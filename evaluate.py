"""
Automated Evaluation and Benchmarking Suite.
Runs scripted multi-turn conversations across all flowchart lanes and reports pass rates.
Generates comprehensive evaluation metrics and sample transcripts.
"""

import sys
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import LeadQualificationAgent
from src.models import QualificationStatus, FlowStep
from src.config import settings


class ConversationBenchmark:
    def __init__(self, name: str, lane: str, turns: List[str], expected_status: QualificationStatus, checks: List[Callable[[LeadQualificationAgent, str, List[str]], bool]], description: str):
        self.name = name
        self.lane = lane
        self.turns = turns
        self.expected_status = expected_status
        self.checks = checks
        self.description = description


def run_evaluations() -> Dict[str, Any]:
    agent = LeadQualificationAgent()
    transcripts_dir = PROJECT_ROOT / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        # 1. Lane A: Qualified New-Build
        ConversationBenchmark(
            name="Lane A - Qualified New-Build",
            lane="A",
            turns=[
                "Hello, I am Sarah",
                "Buy",
                "No",
                "New-build",
                "£1.5m",
                "SW3 3RB",
                "07123 456789",
                "sarah@example.com",
                "No, that's everything",
            ],
            expected_status=QualificationStatus.QUALIFIED,
            checks=[
                lambda a, sid, replies: a.get_session_state(sid).lead.phone == "07123 456789",
                lambda a, sid, replies: a.get_session_state(sid).lead.email == "sarah@example.com",
                lambda a, sid, replies: len(a.get_session_state(sid).lead.matched_properties) > 0,
            ],
            description="New-build buyer with £1.5M budget in SW3 3RB (passes all checks)",
        ),
        # 2. Lane A: Unqualified M1 (Budget < £1M)
        ConversationBenchmark(
            name="Lane A - Unqualified M1 (Budget < 1M)",
            lane="A",
            turns=[
                "Hi, I'm Oliver",
                "Buy",
                "No",
                "New-build",
                "950k",
            ],
            expected_status=QualificationStatus.UNQUALIFIED_M1,
            checks=[
                lambda a, sid, replies: "under £1 million" in replies[-1],
                lambda a, sid, replies: settings.HEAD_OFFICE_PHONE in replies[-1],
                lambda a, sid, replies: a.get_session_state(sid).lead.phone is None,
            ],
            description="New-build buyer with £950k budget correctly triggers M1 exit",
        ),
        # 3. Lane A: Unqualified M2 (Postcode no new-builds)
        ConversationBenchmark(
            name="Lane A - Unqualified M2 (No NB at Postcode)",
            lane="A",
            turns=[
                "Hi, I'm Charlotte",
                "Buy",
                "No",
                "New-build",
                "£2m",
                "SW10 9RR",  # only resale exists here
            ],
            expected_status=QualificationStatus.UNQUALIFIED_M2,
            checks=[
                lambda a, sid, replies: "postcode you provided" in replies[-1],
                lambda a, sid, replies: settings.HEAD_OFFICE_PHONE in replies[-1],
            ],
            description="New-build buyer in SW10 9RR where only resale exists triggers M2",
        ),
        # 4. Lane B: Qualified Resale Buyer
        ConversationBenchmark(
            name="Lane B - Qualified Resale",
            lane="B",
            turns=[
                "Hello, I'm James",
                "Buy",
                "No",
                "Resale",
                "£750,000",
                "SW3 5WU",
                "07987 654321",
                "james@example.com",
                "No thanks",
            ],
            expected_status=QualificationStatus.QUALIFIED,
            checks=[
                lambda a, sid, replies: a.get_session_state(sid).lead.budget_gbp == 750_000,
                lambda a, sid, replies: a.get_session_state(sid).lead.phone == "07987 654321",
                lambda a, sid, replies: len(a.get_session_state(sid).lead.matched_properties) > 0,
            ],
            description="Resale buyer with £750k budget (no min budget constraint) qualifies",
        ),
        # 5. Lane C: Buyer with Specific Property (Available)
        ConversationBenchmark(
            name="Lane C - Specific Property Available",
            lane="C",
            turns=[
                "Hi, I'm Karen",
                "Buy",
                "Yes",
                "Apartment 1, Aldermoor Place",
                "07456 123456",
                "karen@example.com",
                "No, goodbye",
            ],
            expected_status=QualificationStatus.QUALIFIED,
            checks=[
                lambda a, sid, replies: "available" in replies[3].lower(),
                lambda a, sid, replies: a.get_session_state(sid).lead.phone == "07456 123456",
            ],
            description="Specific property lookup finds available apartment and shares details",
        ),
        # 6. Lane C: Specific Property Sold -> Reroute
        ConversationBenchmark(
            name="Lane C - Specific Property Sold Reroute",
            lane="C",
            turns=[
                "Hi, I'm Liam",
                "Buy",
                "Yes",
                "Townhouse 2, Riverside Quarter",  # status: sold
                "£800k",  # new_build, budget < 1M -> M1
            ],
            expected_status=QualificationStatus.UNQUALIFIED_M1,
            checks=[
                lambda a, sid, replies: "sold" in replies[3].lower() or "under offer" in replies[3].lower(),
                lambda a, sid, replies: "under £1 million" in replies[-1],
            ],
            description="Sold property prompts for similar budget; under £1M triggers M1",
        ),
        # 7. Lane D: Qualified Seller
        ConversationBenchmark(
            name="Lane D - Qualified Seller",
            lane="D",
            turns=[
                "Hello, I want to sell my property, my name is Arthur",
                "KT11 4GR",
                "07555 999888",
                "arthur@example.com",
                "No, that is all",
            ],
            expected_status=QualificationStatus.QUALIFIED,
            checks=[
                lambda a, sid, replies: a.get_session_state(sid).lead.postcode == "KT11 4GR",
                lambda a, sid, replies: a.get_session_state(sid).lead.phone == "07555 999888",
            ],
            description="Seller in covered postcode (KT11 4GR) qualifies and leaves contact info",
        ),
        # 8. Lane D: Unqualified Seller (M2)
        ConversationBenchmark(
            name="Lane D - Unqualified Seller M2",
            lane="D",
            turns=[
                "Hello, I am Frank",
                "Sell",
                "E1 6AN",  # London east end, not in coverage list
            ],
            expected_status=QualificationStatus.UNQUALIFIED_M2,
            checks=[
                lambda a, sid, replies: "postcode you provided" in replies[-1],
                lambda a, sid, replies: a.get_session_state(sid).lead.phone is None,
            ],
            description="Seller outside coverage area triggers M2 message without collecting phone",
        ),
        # 9. Mid-Flow FAQ Interruption & Resumption
        ConversationBenchmark(
            name="Mid-Flow FAQ Question Resumption",
            lane="B",
            turns=[
                "Hi, I am Grace",
                "Buy",
                "No",
                "Resale",
                "What are the Chelsea branch opening hours?",  # Interruption!
                "£1.2m",  # Answering the resumed question
                "SW3 5WU",
                "07123 789456",
                "grace@example.com",
                "No, thanks",
            ],
            expected_status=QualificationStatus.QUALIFIED,
            checks=[
                lambda a, sid, replies: "chelsea branch" in replies[4].lower(),
                lambda a, sid, replies: "sunday" in replies[4].lower(),
                lambda a, sid, replies: a.get_session_state(sid).lead.phone == "07123 789456",
            ],
            description="Visitor asks FAQ question mid-conversation; bot answers and resumes flow",
        ),
    ]

    print("\n" + "=" * 80)
    print(" ASHCOMBE & ROWE ESTATES - AUTOMATED CONVERSATION BENCHMARK SUITE")
    print("=" * 80 + "\n")

    passed_count = 0
    total_count = len(benchmarks)
    results = []

    for b in benchmarks:
        sid = str(uuid.uuid4())
        replies = []
        for user_msg in b.turns:
            reply = agent.chat(sid, user_msg)
            replies.append(reply)

        state = agent.get_session_state(sid)
        status_match = state.qualification_status == b.expected_status
        checks_pass = all(chk(agent, sid, replies) for chk in b.checks)
        test_passed = status_match and checks_pass

        if test_passed:
            passed_count += 1
            res_str = "PASS ✅"
        else:
            res_str = "FAIL ❌"

        results.append({
            "name": b.name,
            "lane": b.lane,
            "passed": test_passed,
            "expected_status": b.expected_status.value,
            "actual_status": state.qualification_status.value,
            "turns_count": len(b.turns),
        })

        print(f"[{res_str}] {b.name}")
        print(f"       Description: {b.description}")
        print(f"       Status: Expected={b.expected_status.value} | Actual={state.qualification_status.value}\n")

        # Save individual transcript (JSON and TXT)
        safe_name = b.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("<", "lt")
        transcript_file = transcripts_dir / f"transcript_{safe_name}.json"
        txt_file = transcripts_dir / f"transcript_{safe_name}.txt"
        try:
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark_name": b.name,
                    "lane": b.lane,
                    "passed": test_passed,
                    "lead_data": state.lead.model_dump(),
                    "history": state.history,
                }, f, indent=2)

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"=== {b.name} ===\n")
                f.write(f"Lane: {b.lane} | Result: {'PASS' if test_passed else 'FAIL'}\n")
                f.write(f"Final Qualification Status: {state.qualification_status.value}\n")
                f.write("-" * 50 + "\n\n")
                for msg in state.history:
                    role = "VISITOR" if msg["role"] == "user" else "BOT"
                    f.write(f"{role}: {msg['content']}\n\n")
        except OSError as e:
            print(f"       [Notice: Could not write transcript file to disk ({e}) - continuing benchmark]")

    pass_rate = (passed_count / total_count) * 100
    print("=" * 80)
    print(f" SUMMARY: {passed_count}/{total_count} Passed ({pass_rate:.1f}% Pass Rate)")
    print("=" * 80 + "\n")

    return {
        "passed": passed_count,
        "total": total_count,
        "pass_rate": pass_rate,
        "results": results,
    }


if __name__ == "__main__":
    run_evaluations()
