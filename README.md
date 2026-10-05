# ResearchForge

ResearchForge is a multi-agent AI research system designed to investigate complex questions by decomposing them into focused research tasks, gathering evidence from the web, cross-analyzing findings, critiquing the evidence, and synthesizing a structured research report.

Instead of relying on a single LLM response, ResearchForge separates the research process into specialized stages so that information can be gathered, analyzed, challenged, and synthesized before the final answer is produced.

## Architecture

```text
User Question
      │
      ▼
┌─────────────┐
│   Planner   │
└──────┬──────┘
       │
       │ 3-4 research tasks
       ▼
┌─────────────────────────────────────┐
│        Parallel Research Scouts     │
│                                     │
│  Scout 1   Scout 2   Scout 3 Scout 4│
│     │         │         │       │    │
│     └─────────┴─────────┴───────┘    │
└──────────────────┬──────────────────┘
                   │
                   ▼
             ┌───────────┐
             │  Analyst  │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │  Critic   │
             └─────┬─────┘
                   │
                   ▼
           ┌────────────────┐
           │   Synthesizer  │
           └───────┬────────┘
                   │
                   ▼
            Final Research Report






## 🚀 Live Demo

Try the deployed application:

**[Launch ResearchForge →](https://researchforge-2v6k2rt4avrhmz36npjt8k.streamlit.app/)**

Source code:

**[View the GitHub Repository →](https://github.com/arvindhvijay2011-wq/ResearchForge)**