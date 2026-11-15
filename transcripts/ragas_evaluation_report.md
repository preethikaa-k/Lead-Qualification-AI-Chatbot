# Ashcombe & Rowe Estates — Ragas FAQ Evaluation Report

- **Total Test Cases**: 12
- **Context Precision (Hit@1)**: 100.0%
- **Faithfulness (Factual Consistency)**: 100.0%
- **Zero-Hallucination Guardrail Rate**: 100.0%
- **Overall Pipeline Accuracy**: 100.0%

## Test Case Breakdown

| ID | Category | Question | Expected ID | Retrieved ID | Hit | Faithfulness |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| TC-01 | hours | Can I pop into the Chelsea branch on a Sunday afternoon? | FAQ-023/FAQ-029 | FAQ-023 | ✅ | 1.0 |
| TC-02 | hours | Are any of your offices open on Sunday other than Chelsea? | FAQ-029 | FAQ-029 | ✅ | 1.0 |
| TC-03 | selling | What commission fee do you charge to sell a residential property? | FAQ-040 | FAQ-040 | ✅ | 1.0 |
| TC-04 | contact | Where is your head office located in London? | FAQ-009/FAQ-004 | FAQ-009 | ✅ | 1.0 |
| TC-05 | branches | Which local branch handles properties in the Kew area? | FAQ-020 | FAQ-020 | ✅ | 1.0 |
| TC-06 | selling | Is a property valuation free of charge? | FAQ-039 | FAQ-039 | ✅ | 1.0 |
| TC-07 | buying | What paperwork and documents do I need to submit an offer? | FAQ-036 | FAQ-036 | ✅ | 1.0 |
| TC-08 | company | Do you handle new-build homes priced under £1 million? | FAQ-006 | FAQ-006 | ✅ | 1.0 |
| TC-09 | general | Do you offer lettings or property management services? | FAQ-044 | FAQ-044 | ✅ | 1.0 |
| TC-10 | company | Are you registered with any professional real estate bodies? | FAQ-007 | FAQ-007 | ✅ | 1.0 |
| TC-11 | out_of_domain | Do you have commercial airplane hangars for sale in Manchester? | DEFLECTION | HEAD_OFFICE_FALLBACK | ✅ | 1.0 |
| TC-12 | out_of_domain | Can I get a discount policy on solar panels from David Rowe? | DEFLECTION | HEAD_OFFICE_FALLBACK | ✅ | 1.0 |
