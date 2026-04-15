import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

class AgentState(TypedDict):
    question: str
    research: str
    summary: str
    final_answer: str

def researcher_agent(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="คุณเป็น AI นักวิจัย หาข้อมูลเชิงลึกและละเอียดเกี่ยวกับคำถาม"),
        HumanMessage(content=f"วิจัยเรื่องนี้: {state['question']}")
    ]
    response = llm.invoke(messages)
    state["research"] = response.content
    return state

def summarizer_agent(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="คุณเป็น AI นักสรุป สรุปข้อมูลให้กระชับและเข้าใจง่าย"),
        HumanMessage(content=f"สรุปข้อมูลนี้: {state['research']}")
    ]
    response = llm.invoke(messages)
    state["summary"] = response.content
    return state

def coordinator_agent(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="คุณเป็น AI coordinator รวบรวมข้อมูลและให้คำตอบสุดท้ายที่ดีที่สุด"),
        HumanMessage(content=f"""
คำถาม: {state['question']}
ข้อมูลวิจัย: {state['research']}
สรุป: {state['summary']}
กรุณาให้คำตอบสุดท้ายที่ครบถ้วน
        """)
    ]
    response = llm.invoke(messages)
    state["final_answer"] = response.content
    return state

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("coordinator", coordinator_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", "coordinator")
workflow.add_edge("coordinator", END)

app_graph = workflow.compile()

st.title("🤖 Multi-Agent AI")

question = st.text_input("ถามคำถาม")

if question:
    with st.spinner("กำลังประมวลผล..."):
        result = app_graph.invoke({"question": question, "research": "", "summary": "", "final_answer": ""})

    with st.expander("🔍 ผลวิจัย (Researcher Agent)"):
        st.write(result["research"])

    with st.expander("📝 สรุป (Summarizer Agent)"):
        st.write(result["summary"])

    st.subheader("✅ คำตอบสุดท้าย")
    st.write(result["final_answer"])