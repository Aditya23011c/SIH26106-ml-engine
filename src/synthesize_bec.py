"""
Template-based synthetic BEC (Business Email Compromise) example generator.
No external API calls -> no rate limits, no quota, instant and free.

Usage:
    python src/synthesize_bec.py --per_template 15
"""

import argparse
import json
import random
import csv
from pathlib import Path

random.seed(42)  # reproducible output for the submission doc

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# ---------- Variable pools (swap/extend freely) ----------

FIRST_NAMES = ["Rajesh", "Priya", "Amit", "Sunita", "Vikram", "Neha", "Arjun",
               "Kavita", "Sanjay", "Anjali", "Rohit", "Deepa", "Karan", "Meera"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Reddy", "Iyer", "Nair", "Kapoor",
              "Mehta", "Joshi", "Choudhary", "Malhotra", "Bansal"]
COMPANIES = ["Nova Industries", "BlueRidge Textiles", "Summit Logistics",
             "Orion Manufacturing", "Crestline Traders", "Vertex Solutions",
             "Meridian Exports", "Falcon Enterprises"]
BANKS = ["HDFC Bank", "ICICI Bank", "Axis Bank", "SBI", "Kotak Mahindra Bank"]
AMOUNTS = ["₹4,85,000", "₹12,40,000", "₹2,75,500", "$18,500", "₹9,60,000",
           "₹1,25,000", "$6,200", "₹22,30,000"]
INVOICE_NOS = [f"INV-{random.randint(10000,99999)}" for _ in range(50)]
URGENCY_WORDS = ["immediately", "before end of day", "urgently, within the hour",
                 "before close of business today", "as soon as possible"]

def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def rand_invoice():
    return f"INV-{random.randint(10000,99999)}"

# ---------- Subtype templates ----------
# Each subtype = list of (subject_template, body_template)

SUBTYPES = {
    "fake_ceo_wire_request": [
        (
            "Quick request - {urgency}",
            "Hi {employee},\n\nI'm currently in a board meeting and can't talk right now. "
            "I need you to process a wire transfer of {amount} to a new vendor account {urgency}. "
            "I'll send the account details separately - please confirm you're available to action this now.\n\n"
            "Regards,\n{sender}\nCEO, {company}"
        ),
        (
            "Confidential - handle personally",
            "Hi {employee},\n\nThis is confidential, please don't discuss it with anyone else on the team yet. "
            "I need {amount} transferred to finalize an acquisition we're working on. "
            "Time-sensitive - can you process this {urgency}?\n\nThanks,\n{sender}"
        ),
    ],
    "vendor_invoice_fraud": [
        (
            "Updated payment details for {invoice}",
            "Dear {employee},\n\nPlease note that our banking details have changed. "
            "For invoice {invoice} (amount {amount}), kindly remit payment to our new account at {bank} "
            "going forward. Let us know once the transfer is complete.\n\n"
            "Best regards,\nAccounts Team, {company}"
        ),
        (
            "Outstanding invoice {invoice} - payment overdue",
            "Hello {employee},\n\nInvoice {invoice} for {amount} remains outstanding. "
            "Please process payment {urgency} to our updated beneficiary account "
            "(details attached) to avoid service disruption.\n\nRegards,\n{sender}\n{company}"
        ),
    ],
    "hr_payroll_redirect": [
        (
            "Update to my direct deposit details",
            "Hi HR Team,\n\nI recently switched banks and need to update my direct deposit information "
            "for payroll. Please update my account to the new details I'll send in a follow-up, "
            "and confirm the change takes effect from the next pay cycle.\n\nThanks,\n{sender}"
        ),
        (
            "Payroll account change request - {urgency}",
            "Hello,\n\nCould you please update my salary account details on file? "
            "My previous account is no longer active. Please confirm {urgency} so my next paycheck "
            "isn't delayed.\n\nRegards,\n{sender}"
        ),
    ],
    "fake_vendor_payment_reminder": [
        (
            "Reminder: {invoice} payment pending",
            "Dear {employee},\n\nThis is a friendly reminder that invoice {invoice} "
            "({amount}) from {company} is still pending payment. Kindly process this {urgency} "
            "using the account details previously shared, and send confirmation once done.\n\n"
            "Regards,\n{sender}"
        ),
        (
            "Second notice - {invoice} overdue",
            "Hi {employee},\n\nWe haven't received payment for {invoice} ({amount}). "
            "Please arrange the transfer {urgency} to avoid late fees and account suspension "
            "with {company}.\n\nThanks,\n{sender}"
        ),
    ],
    "fake_legal_compliance_request": [
        (
            "Time-sensitive legal matter - your action needed",
            "Dear {employee},\n\nI'm assisting {sender} on a confidential legal matter regarding "
            "{company}. We need a payment of {amount} processed {urgency} to settle this before "
            "it escalates. Please treat this as strictly confidential.\n\nRegards,\nLegal Counsel"
        ),
        (
            "Compliance action required - {invoice}",
            "Hello {employee},\n\nOur compliance review flagged an outstanding settlement "
            "({invoice}, {amount}) that must be resolved {urgency} to avoid regulatory penalties "
            "for {company}. Please action this personally.\n\nRegards,\n{sender}"
        ),
    ],
}


def fill_template(subject_t, body_t):
    variables = {
        "employee": random.choice(FIRST_NAMES),
        "sender": rand_name(),
        "company": random.choice(COMPANIES),
        "bank": random.choice(BANKS),
        "amount": random.choice(AMOUNTS),
        "invoice": rand_invoice(),
        "urgency": random.choice(URGENCY_WORDS),
    }
    return subject_t.format(**variables), body_t.format(**variables)


def generate(per_template: int):
    rows = []
    for subtype, templates in SUBTYPES.items():
        for subject_t, body_t in templates:
            for _ in range(per_template):
                subject, body = fill_template(subject_t, body_t)
                rows.append({
                    "subject": subject,
                    "body": body,
                    "label": "bec",
                    "subtype": subtype,
                })
    random.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_template", type=int, default=15,
                         help="How many filled variations per template (default 15)")
    args = parser.parse_args()

    rows = generate(args.per_template)

    out_json = PROCESSED_PATH / "bec_synthetic.json"
    out_csv = PROCESSED_PATH / "bec_synthetic.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "body", "label", "subtype"])
        writer.writeheader()
        writer.writerows(rows)

    # distribution summary
    dist = {}
    for r in rows:
        dist[r["subtype"]] = dist.get(r["subtype"], 0) + 1

    print(f"\nGenerated {len(rows)} synthetic BEC examples")
    print(f"Saved to: {out_csv}\n")
    print("Distribution by subtype:")
    for k, v in dist.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()