"""
Email dataset for the Customer Support Email Triage environment.

30 emails across 3 difficulty tiers (10 per tier):
  - triage_basic:       Clear-cut category + priority (easy)
  - triage_ambiguous:   Ambiguous, multi-signal emails (medium)
  - triage_adversarial: Tricky / adversarial emails requiring nuance (hard)

Each email has:
  - email_id, subject, body, sender_tier, previous_tickets, sentiment_score
  - ground_truth: { category, priority, response_keywords, should_escalate }

Categories: billing | technical | account | general | spam
Priorities:  urgent | high | medium | low
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Task 1 — TRIAGE_BASIC: Clear-cut emails (easy)
# ---------------------------------------------------------------------------
TRIAGE_BASIC_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "basic_001",
        "subject": "I cannot log into my account",
        "body": (
            "Hi, I've been trying to log into my account for the past hour and keep "
            "getting an 'Invalid password' error even though I just reset it. "
            "Please help me regain access as soon as possible."
        ),
        "sender_tier": "standard",
        "previous_tickets": 1,
        "sentiment_score": -0.4,
        "ground_truth": {
            "category": "account",
            "priority": "high",
            "response_keywords": ["password reset", "account access", "verification"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_002",
        "subject": "Incorrect charge on my invoice",
        "body": (
            "Hello, I noticed that my last invoice was $49.99 instead of the $29.99 "
            "I was quoted. I'd like a refund for the overcharge of $20. "
            "Please look into this and credit my account."
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": -0.3,
        "ground_truth": {
            "category": "billing",
            "priority": "medium",
            "response_keywords": ["refund", "invoice", "credit", "overcharge"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_003",
        "subject": "App crashes on startup",
        "body": (
            "Your app keeps crashing immediately after I open it on my iPhone 14. "
            "I've already tried uninstalling and reinstalling. This has been going on "
            "since the last update. Please fix this or tell me how to downgrade."
        ),
        "sender_tier": "trial",
        "previous_tickets": 0,
        "sentiment_score": -0.5,
        "ground_truth": {
            "category": "technical",
            "priority": "medium",
            "response_keywords": ["crash", "reinstall", "bug report", "update", "iOS"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_004",
        "subject": "How do I export my data?",
        "body": (
            "Hi there! I'd love to know how I can export all my data from the platform. "
            "Do you have a CSV export option? Looking forward to your reply, thanks!"
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": 0.5,
        "ground_truth": {
            "category": "general",
            "priority": "low",
            "response_keywords": ["export", "CSV", "download", "data"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_005",
        "subject": "Congratulations! You've won $1,000,000",
        "body": (
            "Dear valued customer, you have been selected as our lucky winner! "
            "Click here to claim your prize: http://bit.ly/cl41m-pr1z3. "
            "This offer expires in 24 hours. Don't miss out!"
        ),
        "sender_tier": "trial",
        "previous_tickets": 0,
        "sentiment_score": 0.8,
        "ground_truth": {
            "category": "spam",
            "priority": "low",
            "response_keywords": [],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_006",
        "subject": "Cancel my subscription immediately",
        "body": (
            "I want to cancel my subscription effective today. Please confirm the "
            "cancellation and ensure I am NOT billed next month. "
            "Send me a written confirmation email."
        ),
        "sender_tier": "standard",
        "previous_tickets": 2,
        "sentiment_score": -0.6,
        "ground_truth": {
            "category": "billing",
            "priority": "high",
            "response_keywords": ["cancellation", "confirm", "billing stopped", "refund policy"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_007",
        "subject": "Cannot connect to your API",
        "body": (
            "We're a developer integrating your REST API. We're receiving 503 errors "
            "on the /v2/data endpoint since 9 AM UTC today. Our production system "
            "is down because of this. Please advise urgently."
        ),
        "sender_tier": "premium",
        "previous_tickets": 3,
        "sentiment_score": -0.7,
        "ground_truth": {
            "category": "technical",
            "priority": "urgent",
            "response_keywords": ["API", "503", "outage", "status page", "engineering"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "basic_008",
        "subject": "Update my billing address",
        "body": (
            "Hi, I've recently moved and need to update my billing address. "
            "New address: 42 Elm Street, Springfield, IL 62701. Please update "
            "this before my next invoice."
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": 0.1,
        "ground_truth": {
            "category": "billing",
            "priority": "low",
            "response_keywords": ["address updated", "billing", "confirmed"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_009",
        "subject": "How do I add a team member?",
        "body": (
            "Hello! We just expanded our team and I'd like to invite 3 new members to "
            "our workspace. Could you walk me through the invitation process? "
            "Thank you!"
        ),
        "sender_tier": "premium",
        "previous_tickets": 1,
        "sentiment_score": 0.6,
        "ground_truth": {
            "category": "general",
            "priority": "low",
            "response_keywords": ["invite", "team", "workspace", "settings"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "basic_010",
        "subject": "Password reset email not arriving",
        "body": (
            "I requested a password reset 3 times and never received the email. "
            "I've checked my spam folder. My email is user@example.com. "
            "Please send the reset link directly or reset my password manually."
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": -0.3,
        "ground_truth": {
            "category": "account",
            "priority": "medium",
            "response_keywords": ["password reset", "email delivery", "manual reset"],
            "should_escalate": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Task 2 — TRIAGE_AMBIGUOUS: Context-dependent emails (medium)
# ---------------------------------------------------------------------------
TRIAGE_AMBIGUOUS_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "ambig_001",
        "subject": "This is unacceptable",
        "body": (
            "Your service has been completely unacceptable this month. "
            "I've had THREE separate issues: first my dashboard wouldn't load, "
            "then I was double-charged, and now my account shows the wrong plan. "
            "I pay for premium and this is what I get?"
        ),
        "sender_tier": "premium",
        "previous_tickets": 5,
        "sentiment_score": -0.9,
        "ground_truth": {
            "category": "billing",  # billing is primary (double-charge + wrong plan)
            "priority": "urgent",
            "response_keywords": ["apologize", "refund", "account review", "escalate", "premium"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_002",
        "subject": "Feature request: dark mode",
        "body": (
            "Hi! Love the product. One thing that would make it perfect is a dark mode. "
            "Also, small bug — the export button sometimes doesn't respond on Firefox. "
            "Could you look into that too?"
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": 0.7,
        "ground_truth": {
            "category": "technical",  # bug report takes precedence over feature request
            "priority": "low",
            "response_keywords": ["bug report", "Firefox", "export", "feature roadmap"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "ambig_003",
        "subject": "RE: RE: RE: Invoice #4821",
        "body": (
            "As I've explained THREE TIMES already, Invoice #4821 is incorrect. "
            "You keep telling me it's been 'escalated' but nothing changes. "
            "If this isn't resolved by end of day I'm disputing the charge with my bank."
        ),
        "sender_tier": "standard",
        "previous_tickets": 7,
        "sentiment_score": -1.0,
        "ground_truth": {
            "category": "billing",
            "priority": "urgent",
            "response_keywords": ["invoice", "resolution", "manager", "dispute", "chargeback"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_004",
        "subject": "Account security concern",
        "body": (
            "I received a login notification from an IP I don't recognize (185.220.101.x). "
            "I changed my password immediately but I'm worried my account was compromised. "
            "Can you check my account activity and tell me if anything was accessed?"
        ),
        "sender_tier": "premium",
        "previous_tickets": 1,
        "sentiment_score": -0.5,
        "ground_truth": {
            "category": "account",
            "priority": "urgent",
            "response_keywords": ["security", "account activity", "IP", "session revoke", "2FA"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_005",
        "subject": "Integration stopped working after your update",
        "body": (
            "Our Zapier integration broke after your update last Tuesday. "
            "The webhook is no longer firing. We process ~500 records/day through "
            "this pipeline. Have you changed your API schema? We need this fixed ASAP."
        ),
        "sender_tier": "premium",
        "previous_tickets": 2,
        "sentiment_score": -0.6,
        "ground_truth": {
            "category": "technical",
            "priority": "urgent",
            "response_keywords": ["webhook", "API", "changelog", "integration", "fix"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_006",
        "subject": "Question about pricing + need to upgrade",
        "body": (
            "Hi, I'm currently on the Starter plan ($19/mo). I want to understand "
            "the difference between the Pro ($49) and Business ($99) plans before "
            "I upgrade. Also, do you offer annual discounts? My team is growing fast."
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": 0.3,
        "ground_truth": {
            "category": "billing",  # pricing/upgrade is billing
            "priority": "medium",
            "response_keywords": ["plan comparison", "annual discount", "upgrade", "Pro vs Business"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "ambig_007",
        "subject": "Not receiving reports",
        "body": (
            "The weekly summary report that used to be emailed to me every Monday "
            "hasn't arrived for the past 3 weeks. I haven't changed any settings. "
            "My email is still the same. Could this be a deliverability issue on your end?"
        ),
        "sender_tier": "standard",
        "previous_tickets": 1,
        "sentiment_score": -0.2,
        "ground_truth": {
            "category": "technical",  # email delivery / system issue
            "priority": "medium",
            "response_keywords": ["report", "email delivery", "settings", "resend", "schedule"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "ambig_008",
        "subject": "Compliance question",
        "body": (
            "We are a healthcare company undergoing a HIPAA audit. Our auditor is asking "
            "whether your platform has a BAA (Business Associate Agreement) available. "
            "Do you offer HIPAA-compliant data storage? This is critical for our compliance."
        ),
        "sender_tier": "premium",
        "previous_tickets": 0,
        "sentiment_score": 0.0,
        "ground_truth": {
            "category": "general",  # legal/compliance falls under general
            "priority": "high",
            "response_keywords": ["HIPAA", "BAA", "compliance", "legal team", "data security"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_009",
        "subject": "Charged after cancellation",
        "body": (
            "I cancelled my subscription on March 1st and confirmed the cancellation. "
            "However, I was charged $49.99 on March 15th. I have the cancellation "
            "confirmation email. This needs to be refunded immediately."
        ),
        "sender_tier": "standard",
        "previous_tickets": 1,
        "sentiment_score": -0.7,
        "ground_truth": {
            "category": "billing",
            "priority": "urgent",
            "response_keywords": ["refund", "post-cancellation charge", "confirmation", "24 hours"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "ambig_010",
        "subject": "New employee onboarding help",
        "body": (
            "We just hired 12 new team members and I need to onboard them all to "
            "our workspace. What's the fastest way to do bulk user invitations? "
            "Also, is there an onboarding guide I can share with them?"
        ),
        "sender_tier": "premium",
        "previous_tickets": 3,
        "sentiment_score": 0.4,
        "ground_truth": {
            "category": "general",
            "priority": "medium",
            "response_keywords": ["bulk invite", "onboarding guide", "CSV import", "team setup"],
            "should_escalate": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Task 3 — TRIAGE_ADVERSARIAL: Tricky / adversarial emails (hard)
# ---------------------------------------------------------------------------
TRIAGE_ADVERSARIAL_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "adv_001",
        "subject": "Urgent account issue",
        "body": (
            "URGENT!!! I need help RIGHT NOW. My account is broken and I've lost "
            "all my data!!! This is a DISASTER for my business!!! "
            "I need someone to call me immediately at +1-555-0000."
        ),
        "sender_tier": "trial",  # trial user — urgency is manufactured, not genuine premium
        "previous_tickets": 12,  # serial complainer
        "sentiment_score": -1.0,
        "ground_truth": {
            "category": "account",  # despite drama, it's an account issue
            "priority": "medium",   # trial + serial complainer = medium, not urgent
            "response_keywords": ["account status", "data check", "follow up", "calm"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_002",
        "subject": "Invoice attached for services rendered",
        "body": (
            "Please find attached Invoice #9920 for consulting services rendered "
            "to your company last month ($2,500). Payment is due within 30 days. "
            "Bank transfer details are in the attached PDF."
        ),
        "sender_tier": "trial",
        "previous_tickets": 0,
        "sentiment_score": 0.2,
        "ground_truth": {
            "category": "spam",    # vendor spam masquerading as invoice
            "priority": "low",
            "response_keywords": [],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_003",
        "subject": "I love your product! Quick billing question",
        "body": (
            "Your platform has saved us SO much time — we've recommended it to "
            "5 other companies! Just a tiny thing: I notice my card was charged "
            "$99 instead of $49. Could be a mistake? No rush, just wanted to flag it."
        ),
        "sender_tier": "premium",
        "previous_tickets": 0,
        "sentiment_score": 0.9,
        "ground_truth": {
            "category": "billing",   # positive tone doesn't change the category
            "priority": "high",      # overcharge from premium user = high, not low-priority
            "response_keywords": ["overcharge", "refund", "invoice", "thank you"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_004",
        "subject": "Security vulnerability report",
        "body": (
            "Hi, I'm a security researcher and I've found a potential SQL injection "
            "vulnerability in your login endpoint. I've documented the steps to "
            "reproduce. Please forward this to your security team immediately "
            "before I publish my findings in 72 hours."
        ),
        "sender_tier": "trial",
        "previous_tickets": 0,
        "sentiment_score": 0.0,
        "ground_truth": {
            "category": "technical",   # security = technical, and must escalate
            "priority": "urgent",
            "response_keywords": ["security team", "responsible disclosure", "acknowledge", "CVE"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "adv_005",
        "subject": "RE: Your response was completely unhelpful",
        "body": (
            "The reply I got from your support team yesterday was a copy-paste "
            "response that didn't address my issue at all. My question was about "
            "whether the SSO integration supports SAML 2.0 — not about general "
            "login troubleshooting. Please actually read the ticket before replying."
        ),
        "sender_tier": "premium",
        "previous_tickets": 4,
        "sentiment_score": -0.8,
        "ground_truth": {
            "category": "technical",    # SSO / SAML = technical
            "priority": "high",
            "response_keywords": ["SAML 2.0", "SSO", "apologize", "specific answer", "integration"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_006",
        "subject": "Hi! Just checking in 😊",
        "body": (
            "Hey team! Just wanted to say hi and see if there are any new features "
            "coming out soon? Also, we noticed our API rate limit seems lower today "
            "— is there maintenance happening? Our scripts slowed down this morning."
        ),
        "sender_tier": "premium",
        "previous_tickets": 2,
        "sentiment_score": 0.8,
        "ground_truth": {
            "category": "technical",   # API slowdown is a technical issue buried in pleasantries
            "priority": "high",
            "response_keywords": ["rate limit", "API", "maintenance", "status page"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_007",
        "subject": "Legal notice regarding data breach",
        "body": (
            "We have reason to believe our customer data was exposed through your "
            "platform. We are engaging our legal team and expect a formal response "
            "from your Data Protection Officer within 72 hours as required by GDPR "
            "Article 33. Please acknowledge receipt of this notice."
        ),
        "sender_tier": "premium",
        "previous_tickets": 1,
        "sentiment_score": -0.5,
        "ground_truth": {
            "category": "account",    # data breach = account/security, not just technical
            "priority": "urgent",
            "response_keywords": ["DPO", "GDPR", "acknowledge", "legal", "investigation"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "adv_008",
        "subject": "Free trial expired but I don't want to pay",
        "body": (
            "My free trial ended yesterday. Honestly I don't think the product is "
            "worth the price point. Is there any way to extend the trial a bit longer? "
            "I'm comparing you with Competitor X which is half the price."
        ),
        "sender_tier": "trial",
        "previous_tickets": 0,
        "sentiment_score": -0.1,
        "ground_truth": {
            "category": "billing",    # trial extension = billing decision
            "priority": "medium",
            "response_keywords": ["trial extension", "pricing", "value", "discount offer"],
            "should_escalate": False,
        },
    },
    {
        "email_id": "adv_009",
        "subject": "Account deletion request",
        "body": (
            "Please delete my account and all associated data immediately in "
            "accordance with GDPR Article 17 (right to erasure). This includes "
            "all backups and logs. I want a written confirmation that deletion "
            "is complete. Do not contact me for any reason after this."
        ),
        "sender_tier": "standard",
        "previous_tickets": 0,
        "sentiment_score": -0.3,
        "ground_truth": {
            "category": "account",
            "priority": "high",
            "response_keywords": ["GDPR", "data deletion", "right to erasure", "confirmation", "30 days"],
            "should_escalate": True,
        },
    },
    {
        "email_id": "adv_010",
        "subject": "Payment failed but I can still access?",
        "body": (
            "I got an email saying my payment failed but my account is still active. "
            "I've updated my card now. Will I be charged the overdue amount automatically? "
            "Or do I need to do something? Also, will there be any late fees?"
        ),
        "sender_tier": "standard",
        "previous_tickets": 1,
        "sentiment_score": 0.0,
        "ground_truth": {
            "category": "billing",
            "priority": "medium",
            "response_keywords": ["payment retry", "late fees", "account status", "card updated"],
            "should_escalate": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Master lookup
# ---------------------------------------------------------------------------
EMAIL_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    "triage_basic": TRIAGE_BASIC_EMAILS,
    "triage_ambiguous": TRIAGE_AMBIGUOUS_EMAILS,
    "triage_adversarial": TRIAGE_ADVERSARIAL_EMAILS,
}

VALID_TASK_IDS = list(EMAIL_DATASETS.keys())


def get_emails_for_task(task_id: str) -> List[Dict[str, Any]]:
    """Return the list of emails for a given task ID."""
    if task_id not in EMAIL_DATASETS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASK_IDS}")
    return EMAIL_DATASETS[task_id]
