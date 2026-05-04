from __future__ import annotations

import streamlit as st

from clinic_agents import PediatricClinicWorkflow, SAMPLE_MESSAGES


st.set_page_config(page_title="Pediatric Message Triage", layout="wide")

if "workflow" not in st.session_state:
    st.session_state.workflow = PediatricClinicWorkflow()
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Pediatric Outpatient Message Triage")
st.caption("Multi-Agent Workflow: Intake → Classification → Risk Scoring → Routing → Response → Metrics")

st.info(
    "Safety design: moderate and high-risk messages are not auto-sent. They are routed for staff or provider review."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Incoming Message")
    selected_sample = st.selectbox(
        "Try a sample message / edge case",
        [""] + SAMPLE_MESSAGES,
        index=0,
    )
    message = st.text_area(
        "Patient message",
        value=selected_sample,
        height=140,
        placeholder="Paste or type a patient portal message...",
    )

    submit = st.button("Run multi-agent triage", type="primary")
    batch_submit = st.button("Run sample batch")

if submit and message.strip():
    processed = st.session_state.workflow.process_message(message)
    st.session_state.history.insert(0, processed)

if batch_submit:
    processed_batch = st.session_state.workflow.process_batch(SAMPLE_MESSAGES)
    st.session_state.history = list(reversed(processed_batch)) + st.session_state.history

with right:
    st.subheader("Agent Output")
    if st.session_state.history:
        latest = st.session_state.history[0]
        top_cols = st.columns(3)
        top_cols[0].metric("Classification", latest.classification.category.value)
        top_cols[1].metric("Confidence", f"{latest.classification.confidence:.0%}")
        top_cols[2].metric("Risk level", latest.routing.risk_level.value)

        if latest.routing.risk_level.value == "High":
            st.error(latest.routing.status_badge)
        elif latest.routing.risk_level.value == "Moderate":
            st.warning(latest.routing.status_badge)
        else:
            st.success(latest.routing.status_badge)

        st.write("**Assigned role**")
        st.write(latest.routing.assigned_role)
        st.write("**Route**")
        st.write(latest.routing.route_to)
        st.write("**Action**")
        st.write(latest.routing.action)

        st.write("**Why this classification?**")
        st.caption(latest.classification.why_classified)
        st.write("**Risk rationale**")
        st.caption(latest.risk_assessment.safety_rationale)

        if latest.risk_assessment.escalation_triggers:
            st.write("**Escalation triggers detected**")
            for trigger in latest.risk_assessment.escalation_triggers:
                st.write(f"- {trigger}")
        else:
            st.write("**Escalation triggers detected**")
            st.write("- None detected")

        st.write("**Generated response draft**")
        st.info(latest.routing.response_draft)

        review_cols = st.columns(2)
        review_cols[0].write("**Auto-send allowed**")
        review_cols[0].write("Yes" if latest.routing.auto_send_allowed else "No")
        review_cols[1].write("**Provider review required**")
        review_cols[1].write("Yes" if latest.routing.requires_provider_review else "No")

        if not latest.routing.auto_send_allowed:
            st.warning(latest.routing.safety_note)
        else:
            st.success(latest.routing.safety_note)
    else:
        st.info("Process a message to see the coordinated agent output.")

st.divider()

metrics = st.session_state.workflow.metrics_summary()
st.subheader("Business Value Snapshot")
metric_cols = st.columns(6)
metric_cols[0].metric("Messages processed", metrics["total_messages_processed"])
metric_cols[1].metric("Avg turnaround", f"{metrics['average_turnaround_seconds']} sec")
metric_cols[2].metric("Urgent flags", metrics["urgent_messages_flagged"])
metric_cols[3].metric("Needs review", metrics["review_required_count"])
metric_cols[4].metric("Auto-draft %", f"{metrics['percent_auto_draft']}%")
metric_cols[5].metric("Est. admin time saved", f"{metrics['estimated_admin_minutes_saved']} min")

with st.expander("Time-savings assumptions"):
    st.write(
        f"Assumption: manual handling averages {st.session_state.workflow.metrics_agent.ESTIMATED_MANUAL_MINUTES_PER_MESSAGE} minutes per message. "
        f"AI-assisted triage/review averages {st.session_state.workflow.metrics_agent.ESTIMATED_AI_REVIEW_MINUTES_PER_MESSAGE} minute per message. "
        "These values are for demo purposes only and should be replaced with real clinic baseline data."
    )
    st.write(f"Estimated manual minutes without AI: {metrics['estimated_manual_minutes_without_ai']}")
    st.write(f"Estimated minutes with AI assist: {metrics['estimated_minutes_with_ai_assist']}")

chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("Category Counts")
    st.bar_chart(metrics["count_by_category"])
with chart_cols[1]:
    st.subheader("Risk Level Counts")
    st.bar_chart(metrics["count_by_risk_level"])

st.subheader("Assigned Role Counts")
st.bar_chart(metrics["count_by_role"])

st.subheader("Processing History")
if st.session_state.history:
    st.dataframe(
        [
            {
                "ID": item.message_id,
                "Received": item.intake.received_at.isoformat(timespec="seconds"),
                "Message": item.intake.original_text,
                "Category": item.classification.category.value,
                "Confidence": f"{item.classification.confidence:.0%}",
                "Risk": item.routing.risk_level.value,
                "Triggers": "; ".join(item.risk_assessment.escalation_triggers) or "None",
                "Assigned role": item.routing.assigned_role,
                "Route": item.routing.route_to,
                "Auto-send": item.routing.auto_send_allowed,
                "Provider review": item.routing.requires_provider_review,
                "Turnaround sec": round(item.turnaround_seconds, 3),
            }
            for item in st.session_state.history
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No messages processed yet.")

with st.expander("Real-world scaling notes"):
    st.write(
        "In a real clinic, this workflow would need EHR integration, authenticated patient messaging, "
        "secure APIs, audit logging, role-based access control, encryption, HIPAA-compliant vendors, "
        "and human review for clinical content before any response is sent to a family."
    )
