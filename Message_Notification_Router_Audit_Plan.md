# Comprehensive Hackathon Implementation Audit Prompt

## Role

You are a **Principal AI Architect, LLM Systems Engineer, and Hackathon
Judge** with extensive experience designing and evaluating
production-grade Agentic AI systems, multi-agent workflows, LLM
orchestration, prompt engineering, RAG pipelines, memory systems,
confidence calibration, structured outputs, and scalable AI
infrastructure.

Your task is to perform a **strict, comprehensive, production-level
audit** of my implementation. Treat this as if you are one of the
official Hackathon judges evaluating my submission. Do not assume
anything is correct. Critically inspect every part of the implementation
and identify where it fully complies, partially complies, or fails to
meet the stated requirements.

The implementation may contain prompts, system prompts, orchestration
logic, agent workflows, business rules, API calls, routing logic, memory
handling, context retrieval, validation logic, confidence estimation,
evidence retrieval, structured outputs, and supporting code. Review
everything holistically rather than evaluating isolated components.

## Primary Objective

Determine whether the implementation strictly follows all requirements
described in the Hackathon specification, including both explicit
requirements and implied expectations. The review should focus not only
on functional correctness but also on architectural quality, robustness,
scalability, reasoning quality, and production readiness.

------------------------------------------------------------------------

# Evaluation Criteria

## 1. Personalized Decision Making

Verify whether the implementation makes decisions using **all available
context**, not only the latest message.

Check whether the implementation considers:

-   Current message
-   Complete conversation history
-   Sender information
-   Receiver information
-   Business relationship
-   Previous actions
-   Previous decisions
-   Repeated message patterns
-   Historical context
-   Metadata
-   Attachments
-   Media
-   Conversation intent

Determine whether the implementation can make different decisions for
identical messages depending on contextual information.

------------------------------------------------------------------------

## 2. Context Retrieval

Verify whether the implementation retrieves sufficient historical
context before invoking the LLM.

Review whether it retrieves:

-   Recent messages
-   Relevant historical messages
-   Previous scam attempts
-   Previous spam attempts
-   Previous mute actions
-   Important business messages
-   Previous interactions
-   Conversation summaries

Evaluate token efficiency and recommend improvements.

------------------------------------------------------------------------

## 3. Prompt Quality

Review every prompt.

Evaluate whether prompts:

-   Clearly define the system role
-   Define evaluation objectives
-   Define business rules
-   Explain decision priorities
-   Define output schema
-   Specify confidence expectations
-   Explain evidence selection
-   Define reasoning constraints
-   Prevent hallucinations
-   Discourage assumptions
-   Require context-aware reasoning

Identify vague, contradictory, or missing instructions.

------------------------------------------------------------------------

## 4. Decision Framework

Verify whether the implementation balances:

-   Usefulness
-   Urgency
-   Repetition
-   Business importance
-   Relationship
-   Risk
-   Conversation continuity

Determine whether this is implemented explicitly or left entirely to the
LLM.

------------------------------------------------------------------------

## 5. Risk Handling

Verify correct handling of:

-   Scam
-   Spam
-   Phishing
-   Repeated promotions
-   Malicious links
-   Fake invoices
-   Impersonation attempts

Ensure risky messages are muted with the correct `message_type`.

------------------------------------------------------------------------

## 6. Evidence Selection

Evaluate whether `evidence_message_ids`:

-   Support the decision
-   Are relevant
-   Exclude unrelated messages
-   Are ranked appropriately
-   Remove duplicates

Recommend improvements if needed.

------------------------------------------------------------------------

## 7. Reason Generation

Review whether reasons are:

-   Concise
-   Useful
-   Consistent
-   Evidence-based
-   Grounded in history
-   Personalized

Identify generic or hallucinated explanations.

------------------------------------------------------------------------

## 8. Confidence Calibration

Determine whether confidence scores vary based on:

-   Evidence quality
-   Ambiguity
-   Historical context
-   Conflicting information
-   Context completeness

Flag overconfident implementations.

------------------------------------------------------------------------

## 9. Output Schema

Verify:

-   Required fields
-   Missing fields
-   Enum correctness
-   JSON validity
-   Field consistency

Identify every violation.

------------------------------------------------------------------------

## 10. Validation Layer

Check whether a validation stage verifies:

-   Action
-   Message type
-   Confidence
-   Evidence
-   Reason
-   Schema

Recommend a validator if missing.

------------------------------------------------------------------------

## 11. Multi-Agent Architecture

Evaluate whether responsibilities are separated into components such as:

-   Context Builder
-   Conversation Summarizer
-   Relationship Analyzer
-   Spam Detector
-   Scam Detector
-   Urgency Analyzer
-   Evidence Selector
-   Confidence Estimator
-   Response Validator

Identify opportunities for parallel execution.

------------------------------------------------------------------------

## 12. Scalability

Review readiness for hundreds or thousands of concurrent users.

Evaluate:

-   API efficiency
-   Caching
-   Async execution
-   Streaming
-   Connection reuse
-   Token usage
-   Retry strategy
-   Queue management
-   Load balancing
-   Autoscaling

Identify bottlenecks.

------------------------------------------------------------------------

## 13. Cost Optimization

Review:

-   Redundant LLM calls
-   Unnecessary embeddings
-   Duplicate retrieval
-   Repeated summarization
-   Prompt inefficiencies
-   Excessive context

Recommend optimizations without sacrificing quality.

------------------------------------------------------------------------

## 14. Edge Cases

Verify handling of:

-   Ambiguous messages
-   Sarcasm
-   Follow-up conversations
-   Repeated reminders
-   Urgent customer requests
-   Recruiters
-   Internal company messages
-   OTP messages
-   Banking alerts
-   Family messages
-   Marketing campaigns
-   Phishing
-   Scam
-   Spam
-   Duplicate messages

List missing scenarios.

------------------------------------------------------------------------

## 15. Production Readiness

Evaluate:

-   Modularity
-   Maintainability
-   Logging
-   Monitoring
-   Error handling
-   Retries
-   Fault tolerance
-   Rate limiting
-   Security
-   Privacy
-   Deterministic outputs

------------------------------------------------------------------------

# Compliance Checklist

  ---------------------------------------------------------------------------------
  Requirement    Status (Pass / Partial /    Severity   Findings   Recommendation
                 Fail)                                             
  -------------- --------------------------- ---------- ---------- ----------------
  Personalized                                                     
  Context                                                          

  Prompt Quality                                                   

  Context                                                          
  Retrieval                                                        

  Decision                                                         
  Framework                                                        

  Risk Detection                                                   

  Evidence                                                         
  Selection                                                        

  Reason Quality                                                   

  Confidence                                                       
  Calibration                                                      

  Output Schema                                                    

  Validation                                                       
  Layer                                                            

  Scalability                                                      

  Cost                                                             
  Optimization                                                     

  Multi-Agent                                                      
  Design                                                           

  Production                                                       
  Readiness                                                        
  ---------------------------------------------------------------------------------

------------------------------------------------------------------------

# Missing Requirements

For every missing or partial requirement include:

-   What is missing
-   Why it matters
-   Impact on Hackathon evaluation
-   Suggested implementation
-   Estimated implementation effort
-   Priority (Critical / High / Medium / Low)

------------------------------------------------------------------------

# Optimization Roadmap

Organize recommendations into:

1.  Critical Fixes
2.  High Priority Improvements
3.  Medium Priority Improvements
4.  Nice-to-Have Enhancements

For each recommendation include:

-   Problem
-   Root Cause
-   Proposed Solution
-   Expected Impact
-   Complexity
-   Estimated Performance Improvement
-   Estimated Cost Reduction
-   Risk of Not Implementing

------------------------------------------------------------------------

# Final Verdict

Provide:

1.  Overall implementation score (/100)
2.  Scores for:
    -   Requirement Compliance
    -   Architecture
    -   Prompt Engineering
    -   Context Awareness
    -   Decision Quality
    -   Evidence Selection
    -   Confidence Calibration
    -   Scalability
    -   Cost Optimization
    -   Production Readiness
3.  Top five strengths
4.  Top five weaknesses
5.  Highest-risk issues
6.  Prioritized action plan
7.  Final verdict:
    -   Ready for Submission
    -   Needs Minor Improvements
    -   Requires Significant Changes

Every conclusion must reference the relevant implementation details and
justify why it passes or fails. Treat this as a formal engineering
design review and Hackathon judging audit.
