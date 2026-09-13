# Intent Taxonomy — AmazonHelp Support Agent

Defined from manual review of 130 real customer↔AmazonHelp reply pairs 
(30-row initial sample + 100-row stress-test sample, random_state=42) 
drawn from data/amazon_threads.csv (123,569 English-only matched pairs).

## Intents

1. **Delivery Delay / Non-Delivery**
   Customer's order is late, stuck, marked delivered but not received, 
   or delivery window was missed. Most frequent category observed 
   (~35-40% of sampled rows) — see decision log for baseline implications.

2. **Wrong / Missing / Damaged Item**
   Customer received the wrong product, wrong size/variant, a damaged 
   item, or part of their order is missing.

3. **Refund / Billing / Charge Dispute**
   Customer disputes a charge, requests a refund, reports an unexpected 
   or recurring charge, or a promised refund hasn't arrived.

4. **Account Access Issue**
   Login/password problems, account closed/suspended unexpectedly, 
   suspected account compromise, or missing order history.

5. **App / Digital Content / Technical Issue**
   Problems with the Amazon app, digital content (Prime Video, Music), 
   Alexa/Echo devices, checkout errors, or website bugs.

6. **Marketplace / Seller / Pricing Issue**
   Complaints about third-party sellers, pricing changes, or confusion 
   about who (Amazon vs. seller) is responsible for an issue.

7. **General Inquiry / Feedback**
   Vague complaints without a clear actionable request, general 
   questions, positive feedback/gratitude, or feedback not tied to 
   a specific transaction.

## Cross-Cutting Signal (not an intent): Explicit Escalation Request

Independent of intent, some messages explicitly demand human contact 
("call me," "let me speak to someone," "callback"). This is tracked 
as a separate boolean signal feeding into the AUTO-HANDLE vs ESCALATE 
decision (see Phase 4 escalation policy), not as its own intent, 
since it can co-occur with any of the 7 categories above.

## What was deliberately NOT split out

- Wrong/missing/damaged items were kept as one category rather than 
  three, since the downstream resolution path (replacement/refund flow) 
  is the same in the historical data, and splitting further would 
  leave too few examples per class given the 150-250 golden set size.
- Multilingual intents were not built (scope decision — see decision log).