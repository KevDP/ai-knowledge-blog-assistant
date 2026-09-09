"""Weekly usage digest for EVA and the portfolio site.

Answers the question alarms cannot: "is anyone actually using this?"

An alarm is for rare, actionable events that should wake someone up. Usage on a
promotion-driven portfolio legitimately reads zero for days at a time, so an
alarm on it would fire during normal quiet weeks and train the reader to ignore
alarms. Usage is state that changes slowly, so it belongs in a periodic report.

Reads AWS/Lambda, AWS/Bedrock and AWS/CloudFront through GetMetricData, and the
CloudFront Function's own log through Logs Insights. Deliberately reads standard
metrics instead of publishing custom ones.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time

import boto3

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REGION = os.environ.get("AWS_REGION", "us-east-1")
TOPIC_ARN = os.environ["TOPIC_ARN"]
FUNCTION_NAME = os.environ["EVA_FUNCTION_NAME"]
API_ID = os.environ["EVA_API_ID"]
DISTRIBUTION_ID = os.environ.get("CLOUDFRONT_DISTRIBUTION_ID", "")
EDGE_LOG_GROUP = os.environ.get("EDGE_LOG_GROUP", "")
CHAT_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
EMBED_MODEL_ID = os.environ["TITAN_MODEL_ID"]

WINDOW_DAYS = 7

# Bedrock on-demand pricing, USD per 1M tokens. Hardcoded on purpose.
# If Bedrock pricing changes, this is the one place to update.

PRICE_CHAT_INPUT_PER_M = 1.00
PRICE_CHAT_OUTPUT_PER_M = 5.00
PRICE_EMBED_INPUT_PER_M = 0.02

BOT_MARKERS = (
    "bot", "crawler", "spider", "slurp", "curl", "wget", "python",
    "scrapy", "headless", "lighthouse", "pagespeed", "http-client",
    "okhttp", "java/", "go-http", "libwww", "axios", "postman", "scan",
)

# Version strings no real OS or browser ships. Added after the first eight hours
# of edge logging, where BOT_MARKERS alone reported 6.4% automated traffic and
# the actual figure was around 78%: one scanner family was rotating fabricated
# browser strings, eight of them, at exactly 49 requests each. Uniform counts
# across distinct agents is itself a fingerprint no human traffic produces.
#
# These patterns catch that family. They do not catch a scanner that copies a
# real, current user agent, and nothing short of behavioural analysis would.
# AWS sells that as a $200/month tier; the need here is to put an honest number
# in an email, so three regexes is where this stops.
FAKE_UA_PATTERNS = (
    # Real Windows agents are dot-separated ("Windows NT 10.0"), and NT 9 never
    # existed. An underscore after the major version is fabricated.
    re.compile(r"Windows NT \d+_"),
    # macOS reports 10_x_y and Apple freezes the string at 10_15_7. Any other
    # major version is invented.
    re.compile(r"Mac OS X (?!10[._])\d"),
    # Chrome-family browsers have shipped AppleWebKit/537.36 for over a decade.
    # A 5xx build that is not 537.36 while still claiming Chrome is made up.
    re.compile(r"AppleWebKit/5(?!37\.36)\d\d\.\d+.*Chrome/"),
)


def _is_automated(ua: str) -> bool:
    """True when the agent declares itself a bot or carries an impossible version."""
    low = ua.lower()
    if any(m in low for m in BOT_MARKERS):
        return True
    return any(p.search(ua) for p in FAKE_UA_PATTERNS)

INSIGHTS_POLL_SECONDS = 35

cw = boto3.client("cloudwatch", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)


# ─────────────────────────────────────────────────────────────────────────────
# Metric access
# ─────────────────────────────────────────────────────────────────────────────


def _q(qid: str, namespace: str, metric: str, dims: dict[str, str], stat: str) -> dict:
    """Build one GetMetricData query with a full-window period.

    Period equals the whole window so each query returns a single datapoint,
    which keeps the aggregation in CloudWatch instead of in this function.
    """
    return {
        "Id": qid,
        "MetricStat": {
            "Metric": {
                "Namespace": namespace,
                "MetricName": metric,
                "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()],
            },
            "Period": WINDOW_DAYS * 86400,
            "Stat": stat,
        },
    }


def _fetch(queries: list[dict], start: dt.datetime, end: dt.datetime) -> dict[str, float]:
    """Run the queries and flatten to {id: value}, with 0.0 for empty results.

    An empty result means the metric received no data in the window, which for
    Invocations means "nobody asked anything". That is a real answer, not an
    error, so it collapses to 0.0 rather than raising.
    """
    out: dict[str, float] = {}
    resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end)
    for r in resp["MetricDataResults"]:
        vals = r.get("Values") or [0.0]
        out[r["Id"]] = float(vals[0])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Edge log: who is actually hitting the site
# ─────────────────────────────────────────────────────────────────────────────


def _edge_traffic(start: dt.datetime, end: dt.datetime) -> dict | None:
    """Aggregate user agents from the CloudFront Function log.

    Returns None when the log group is unset or does not exist yet, which is the
    normal state until the function has been published and served one request.
    A missing producer degrades this section instead of failing the whole email.

    The user agent is extracted with a regex rather than by JSON field discovery
    because the function writes its JSON through console.log, and CloudFront may
    wrap that line before it reaches CloudWatch. A user agent containing an
    escaped quote will be truncated at that quote; it lands in its own bucket
    and is not worth more parsing than that.
    """
    if not EDGE_LOG_GROUP:
        return None

    query = (
        "fields @message\n"
        '| parse @message /"ua":"(?<ua>[^"]*)"/\n'
        "| filter ispresent(ua)\n"
        "| stats count(*) as n by ua\n"
        "| sort n desc\n"
        "| limit 60"
    )
    try:
        qid = logs.start_query(
            logGroupName=EDGE_LOG_GROUP,
            startTime=int(start.timestamp()),
            endTime=int(end.timestamp()),
            queryString=query,
            limit=10000,
        )["queryId"]
    except logs.exceptions.ResourceNotFoundException:
        return None

    deadline = time.time() + INSIGHTS_POLL_SECONDS
    result = None
    while time.time() < deadline:
        r = logs.get_query_results(queryId=qid)
        if r["status"] == "Complete":
            result = r
            break
        if r["status"] in ("Failed", "Cancelled", "Timeout"):
            return None
        time.sleep(1)
    if result is None:
        logs.stop_query(queryId=qid)
        return None

    rows: list[tuple[str, int]] = []
    for row in result["results"]:
        f = {c["field"]: c["value"] for c in row}
        if "ua" in f and "n" in f:
            rows.append((f["ua"], int(f["n"])))

    total = sum(n for _, n in rows)
    bots = sum(n for ua, n in rows if _is_automated(ua))
    return {"rows": rows, "total": total, "bots": bots}


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _build_report(start: dt.datetime, end: dt.datetime) -> str:
    queries = [
        _q("inv", "AWS/Lambda", "Invocations", {"FunctionName": FUNCTION_NAME}, "Sum"),
        _q("err", "AWS/Lambda", "Errors", {"FunctionName": FUNCTION_NAME}, "Sum"),
        _q("dur", "AWS/Lambda", "Duration", {"FunctionName": FUNCTION_NAME}, "Average"),
        _q("api", "AWS/ApiGateway", "Count", {"ApiId": API_ID}, "Sum"),
        _q("itok", "AWS/Bedrock", "InputTokenCount", {"ModelId": CHAT_MODEL_ID}, "Sum"),
        _q("otok", "AWS/Bedrock", "OutputTokenCount", {"ModelId": CHAT_MODEL_ID}, "Sum"),
        _q("etok", "AWS/Bedrock", "InputTokenCount", {"ModelId": EMBED_MODEL_ID}, "Sum"),
    ]
    if DISTRIBUTION_ID:
        cf_dims = {"DistributionId": DISTRIBUTION_ID, "Region": "Global"}
        queries += [
            _q("req", "AWS/CloudFront", "Requests", cf_dims, "Sum"),
            _q("e4", "AWS/CloudFront", "4xxErrorRate", cf_dims, "Average"),
            _q("e5", "AWS/CloudFront", "5xxErrorRate", cf_dims, "Average"),
        ]

    m = _fetch(queries, start, end)

    chat_cost = (
        m["itok"] / 1_000_000 * PRICE_CHAT_INPUT_PER_M
        + m["otok"] / 1_000_000 * PRICE_CHAT_OUTPUT_PER_M
    )
    embed_cost = m["etok"] / 1_000_000 * PRICE_EMBED_INPUT_PER_M
    total_cost = chat_cost + embed_cost
    per_query = total_cost / m["inv"] if m["inv"] else 0.0
    avg_in = m["itok"] / m["inv"] if m["inv"] else 0.0

    lines = [
        f"EVA weekly digest: {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        "",
        "EVA",
        f"  questions answered   {m['inv']:.0f}",
        f"  api gateway requests {m['api']:.0f}  (extra ones are CORS preflights)",
        f"  errors               {m['err']:.0f}",
        f"  avg duration         {m['dur']:.0f} ms",
        "",
        "Cost",
        f"  bedrock total        ${total_cost:.4f}",
        f"  per question         ${per_query:.4f}",
        f"  avg input tokens     {avg_in:.0f}",
    ]

    if DISTRIBUTION_ID:
        lines += [
            "",
            "Site",
            f"  requests             {m['req']:.0f}",
            f"  4xx rate             {m['e4']:.1f}%   (high means bot scanning)",
            f"  5xx rate             {m['e5']:.2f}%",
        ]

    edge = _edge_traffic(start, end)
    if edge and edge["total"]:
        bot_share = edge["bots"] / edge["total"] * 100
        lines += ["", "Traffic composition (edge function log)"]
        if DISTRIBUTION_ID and m.get("req"):


            captured = edge["total"] / m["req"] * 100
            lines.append(f"  log lines            {edge['total']:,} of {m['req']:,.0f} requests ({captured:.1f}% captured)")
        else:
            lines.append(f"  log lines            {edge['total']:,}")
        lines += [
            f"  automated traffic    at least {bot_share:.1f}%",
            "  top agents",
        ]
        for ua, n in edge["rows"][:8]:
            label = ua if len(ua) <= 68 else ua[:65] + "..."
            lines.append(f"    {n:>7,}  {label or '(no user agent)'}")
        lines.append("  Counted per request, not per visit. A page view pulls its assets")
        lines.append("  through here too (~2.7 lines), while a scanner that probes one path")
        lines.append("  and leaves costs 1. So both numbers understate bots per visitor, and")
        lines.append("  'at least' is literal: an agent copying a browser string reads as")
        lines.append("  human. The share can only move up.")
    elif EDGE_LOG_GROUP:
        lines += [
            "",
            "Traffic composition: no edge log data this week.",
            "Either the CloudFront Function was never published to LIVE, or it",
            "served no requests. Saving the function is not publishing it.",
        ]

    if m["inv"] == 0:
        lines += [
            "",
            "No questions this week. For a promotion-driven portfolio that is",
            "expected, not a failure. If it repeats for several weeks, the fix is",
            "distribution, not more instrumentation.",
        ]

    return "\n".join(lines)


def lambda_handler(event, context):
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=WINDOW_DAYS)
    body = _build_report(start, end)
    sns.publish(
        TopicArn=TOPIC_ARN,
        Subject=f"EVA weekly digest {start:%b %d} to {end:%b %d}",
        Message=body,
    )
    return {"ok": True, "window_days": WINDOW_DAYS}
