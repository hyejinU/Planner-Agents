"""
Chainlit Frontend for E-commerce Text2SQL + Agentic Speculation
"""

import json
import chainlit as cl
from text2sql_agent import process_question_stream, generate_graph_visualization


##Generate workflow diagram once at module load (optional)
##Uncomment the lines below if you want to generate the diagram:
try:
    workflow_diagram_path = generate_graph_visualization("text2sql_workflow.png")
    if workflow_diagram_path:
        print(f"✅ Workflow diagram generated: {workflow_diagram_path}")
except Exception as e:
    print(f"⚠️ Warning: Could not generate workflow diagram: {e}")

# -------------------------------
# Chat 시작 인트로 메시지
# -------------------------------

@cl.on_chat_start
async def start():
    """Initialize the chat session"""

    await cl.Message(
        content=(
            "👋 Welcome to the Text2SQL E-commerce Assistant!\n\n"
            "I can help you query the e-commerce database using natural language. "
            "Just ask me questions about:\n"
            "- Orders and their status\n"
            "- Customers and their locations\n"
            "- Products and categories\n"
            "- Payments and transactions\n"
            "- Reviews and ratings\n"
            "- Sellers and their information\n\n"
            "**Example questions:**\n"
            "- How many orders were delivered?\n"
            "- What are the top 5 product categories by sales?\n"
            "- Show me orders from São Paulo\n"
            "- What's the average review score?\n"
            "- Which sellers have the most orders?\n"
            "- Try three different coupon strategies and compare revenue\n\n"
            "Go ahead and ask me anything! 🚀"
        )
    ).send()


# -------------------------------
# 메인 메시지 핸들러
# -------------------------------

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with debugging visualization"""

    user_question = message.content

    # 메인 워크플로 Step (전체 에이전트 실행을 감싸는 상위 Step)
    async with cl.Step(name="🤖 Agent Workflow", type="llm") as workflow_step:

        node_steps = {}   # node_name -> Chainlit Step
        final_state = None

        # 노드 이름 → 예쁜 표시 이름 매핑
        node_display_names = {
            "guardrails_agent": "🛡 Guardrails (Scope Check)",
            "router_agent": "🚦 Router (Intent Classification)",

            "sql_agent_read_only": "📝 Generate SQL (Read-Only)",
            "execute_sql_read_only_agent": "⚙️ Execute SQL (Read-Only)",

            "experiment_planner_agent": "🧠 Experiment Planner",
            "branch_world_creator_agent": "🌱 Create Branch Worlds",
            "sql_agent_experiment": "📝 Generate SQL (Branches)",
            "execute_sql_agent": "⚙️ Execute SQL (Branches)",
            "error_agent": "🔧 Fix SQL Error",
            "evaluate_agent": "🏆 Evaluate Strategies",

            "branch_control_agent": "📌 Commit/Rollback Branch",
            "sql_agent_schema": "🛠 SQL (Schema Change)",
        }

        try:
            # LangGraph 이벤트 스트림 처리
            async for event in process_question_stream(user_question):
                event_type = event.get("type")

                # ---------------- node_start ----------------
                if event_type == "node_start":
                    node_name = event["node"]
                    display_name = node_display_names.get(node_name, node_name)

                    node_step = cl.Step(
                        name=display_name,
                        type="tool",
                        parent_id=workflow_step.id,
                    )
                    await node_step.send()
                    node_steps[node_name] = node_step

                # ---------------- node_end ----------------
                elif event_type == "node_end":
                    node_name = event["node"]
                    output = event.get("output", {}) or {}
                    state = event.get("state", {}) or {}

                    if node_name in node_steps:
                        node_step = node_steps[node_name]
                        output_text = ""

                        # 편의상 state에서 필요한 값들을 읽자 (output보다 state가 더 믿을 만 함)
                        intent = state.get("intent")

                        if node_name == "guardrails_agent":
                            in_scope = state.get("guardrail_in_scope", True)
                            reason = state.get("guardrail_reason", "")
                            status = "IN SCOPE ✅" if in_scope else "OUT OF SCOPE ❌"
                            output_text = (
                                f"**Guardrail Result:** {status}\n"
                                f"**Reason:** {reason}"
                            )

                        elif node_name == "router_agent":
                            intent = state.get("intent", "(unknown)")
                            reason = state.get("router_reason", "")
                            output_text = (
                                f"**Intent:** `{intent}`\n"
                                f"**Reason:** {reason}"
                            )

                        elif node_name == "sql_agent_read_only":
                            sql = state.get("read_only_sql", "")
                            output_text = (
                                "**Generated Read-Only SQL:**\n"
                                f"```sql\n{sql}\n```"
                            )

                        elif node_name == "execute_sql_read_only_agent":
                            msg = state.get("read_only_result_message", "")
                            output_text = (
                                "**Read-Only Query Result:**\n"
                                f"{msg}"
                            )

                        elif node_name == "experiment_planner_agent":
                            plan = state.get("branch_plan", {})
                            branches = plan.get("branches", [])
                            primary = plan.get("primary_metric", "")
                            sec = plan.get("secondary_metrics", [])
                            lines = [
                                f"**Planned {len(branches)} strategies.**",
                                f"- Primary metric: `{primary}`",
                            ]
                            if sec:
                                lines.append(f"- Secondary metrics: {sec}")
                            for b in branches:
                                bid = b.get("branch_id")
                                name = b.get("name", "")
                                hyp = b.get("hypothesis", "")
                                lines.append(f"  - `{bid}`: **{name}** — {hyp}")
                            output_text = "\n".join(lines)

                        elif node_name == "branch_world_creator_agent":
                            plan = state.get("branch_plan", {})
                            b2w = plan.get("branch_to_world", {})
                            lines = ["**Created branch worlds:**"]
                            if not b2w:
                                lines.append("- (no worlds created)")
                            else:
                                for bid, wid in b2w.items():
                                    lines.append(f"- Branch `{bid}` → World `{wid}`")
                            output_text = "\n".join(lines)

                        elif node_name == "sql_agent_experiment":
                            branch_sql = state.get("branch_sql", {})
                            lines = ["**Generated SQL for branches:**"]
                            if not branch_sql:
                                lines.append("- (no SQL generated)")
                            else:
                                for wid, sql_list in branch_sql.items():
                                    lines.append(
                                        f"- World `{wid}`: {len(sql_list)} statements"
                                    )
                            output_text = "\n".join(lines)

                        elif node_name == "execute_sql_agent":
                            failed = state.get("failed_worlds", [])
                            needs_err = state.get("needs_error_handling", False)
                            branch_results = state.get("branch_results") or {}

                            lines = [
                                "**Executed branch SQL (per world):**",
                                f"- Failed worlds: {failed}",
                                f"- Needs error handling (next step = error_agent?): {needs_err}",
                                "",
                            ]

                            if not branch_results:
                                lines.append("_아직 실행된 SQL 로그가 없습니다._")
                            else:
                                for world_id, res in branch_results.items():
                                    lines.append(f"---")
                                    lines.append(f"### 🌍 World `{world_id}`")

                                    status = res.get("status", "ok")
                                    if status != "ok":
                                        lines.append(f"- Status: `{status}`")
                                        if res.get("failure_reason"):
                                            lines.append(f"- Failure reason: {res['failure_reason']}")

                                    sql_log = res.get("sql_log") or []
                                    metrics = res.get("metrics") or {}

                                    # metrics 요약
                                    if metrics:
                                        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                                        lines.append(f"- Metrics: {metric_str}")

                                    if not sql_log:
                                        lines.append("- (실행된 SQL이 없습니다.)")
                                        continue

                                    # world별 실행된 SQL 나열
                                    for i, entry in enumerate(sql_log, start=1):
                                        stmt = entry.get("statement", "")
                                        stmt_disp = stmt
                                        # 너무 길면 잘라주기
                                        if len(stmt_disp) > 400:
                                            stmt_disp = stmt_disp[:400] + "\n...(truncated)"

                                        lines.append(f"\n**[{i}] SQL**")
                                        lines.append(f"```sql\n{stmt_disp}\n```")

                            output_text = "\n".join(lines)

                        elif node_name == "error_agent":
                            last_err = state.get("last_error", "")
                            raw = state.get("error_agent_raw", "")
                            if len(raw) > 400:
                                raw = raw[:400] + "\n...(truncated)"
                            output_text = (
                                f"**Last Error:**\n```\n{last_err}\n```\n\n"
                                f"**Error Agent Output (corrected SQL or IMPOSSIBLE):**\n"
                                f"```sql\n{raw}\n```"
                            )

                        elif node_name == "evaluate_agent":
                            eval_msg = state.get("evaluation_message", "")
                            if len(eval_msg) > 800:
                                disp = eval_msg[:800] + "\n...(truncated)"
                            else:
                                disp = eval_msg
                            output_text = (
                                "**Evaluation Summary:**\n"
                                f"{disp}"
                            )

                        elif node_name == "branch_control_agent":
                            msg = state.get("commit_result_message", "")
                            output_text = (
                                "**Branch Commit/Rollback Result:**\n"
                                f"{msg}"
                            )

                        elif node_name == "sql_agent_schema":
                            branch_sql = state.get("branch_sql", {})
                            world_id = state.get("current_world_id", "main")
                            sql_list = branch_sql.get(world_id, [])
                            lines = [f"**Schema-change SQL for world `{world_id}`:**"]
                            if not sql_list:
                                lines.append("- (no SQL generated)")
                            else:
                                for i, s in enumerate(sql_list, 1):
                                    lines.append(f"--- Statement {i} ---")
                                    lines.append(f"```sql\n{s}\n```")
                            output_text = "\n".join(lines)
                            
                        elif node_name == "decide_graph_need":
                            needs_graph = output.get("needs_graph", False)
                            graph_type = output.get("graph_type", "")
                            if needs_graph:
                                output_text = f"✅ **Graph Needed:** {graph_type.upper()} chart"
                            else:
                                output_text = "ℹ️ **No graph needed** for this query"
                        
                        elif node_name == "generate_graph":
                            has_graph = bool(output.get("graph_json"))
                            if has_graph:
                                output_text = "✅ Graph generated successfully"
                            else:
                                output_text = "⚠️ Graph generation skipped"

                        else:
                            # 기본 포맷: state 전체를 JSON으로 보여주고 싶으면 여기서
                            output_text = f"State snapshot:\n```json\n{json.dumps(state, ensure_ascii=False, indent=2)}\n```"

                        node_step.output = output_text
                        await node_step.update()

                # ---------------- final ----------------
                elif event_type == "final":
                    final_state = event.get("result", {})

                # ---------------- error ----------------
                elif event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    workflow_step.output = f"❌ **Error:** {error_msg}"
                    await workflow_step.update()
                    return

            # 워크플로 Step 완료 표시
            workflow_step.output = "✅ Workflow completed successfully"
            await workflow_step.update()

        except Exception as e:
            workflow_step.output = f"❌ **Unexpected Error:** {str(e)}"
            await workflow_step.update()
            raise

    # ---------------- 최종 사용자 응답 ----------------
    if final_state:
        intent = final_state.get("intent")
        response_content = ""

        if intent == "READ_ONLY":
            # READ_ONLY: 결과 텍스트 그대로 보여줌
            msg = final_state.get("read_only_result_message") or "READ_ONLY 결과가 없습니다."
            response_content = msg

        elif intent in ("EXPERIMENT_START", "SCHEMA_CHANGE"):
            # 1) 평가 요약 + 커밋 결과 먼저 구성
            eval_msg = final_state.get("evaluation_message") or "실험 결과를 생성하지 못했습니다."
            commit_msg = final_state.get("commit_result_message")
            if commit_msg:
                response_content = eval_msg + "\n\n---\n\n" + commit_msg
            else:
                response_content = eval_msg

            # 2) 디버그용: 각 world에서 어떤 SQL이 실행되었는지 요약해서 붙이기
            branch_results = final_state.get("branch_results") or {}
            if branch_results:
                debug_lines = [
                    "",
                    "",
                    "---",
                    "🧪 *디버그: 각 World에서 실행된 SQL 목록*",
                ]
                for world_id, res in branch_results.items():
                    status = res.get("status", "ok")
                    sql_log = res.get("sql_log") or []
                    metrics = res.get("metrics") or {}

                    debug_lines.append(f"\n### 🌍 World `{world_id}`")
                    if status != "ok":
                        debug_lines.append(f"- Status: `{status}`")
                        if res.get("failure_reason"):
                            debug_lines.append(f"- Failure reason: {res['failure_reason']}")
                    # metrics 요약
                    if metrics:
                        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                        debug_lines.append(f"- Metrics: {metric_str}")

                    if not sql_log:
                        debug_lines.append("- (실행된 SQL이 없습니다.)")
                        continue

                    # 이 world에서 실행된 모든 SQL 나열 (길면 잘라서)
                    for i, entry in enumerate(sql_log, start=1):
                        stmt = entry.get("statement", "")
                        stmt_disp = stmt
                        if len(stmt_disp) > 400:
                            stmt_disp = stmt_disp[:400] + "\n...(truncated)"

                        debug_lines.append(f"\n**[{i}] SQL**")
                        debug_lines.append(f"```sql\n{stmt_disp}\n```")

                response_content += "\n".join(debug_lines)


        elif intent == "BRANCH_CONTROL":
            # 커밋/롤백 결과
            commit_msg = final_state.get("commit_result_message") or "커밋/롤백 결과가 없습니다."
            response_content = commit_msg

        elif intent == "OUT_OF_SCOPE":
            # out-of-scope: guardrail_reason 기반
            reason = final_state.get("guardrail_reason", "")
            response_content = "이 질문은 현재 E-commerce 데이터베이스 범위를 벗어납니다."
            if reason:
                response_content += f"\n이유: {reason}"

        else:
            # 그 외 (혹은 intent 미설정) fallback
            response_content = (
                "요청을 처리했지만, 응답을 생성할 수 없습니다.\n"
                f"(intent: {intent})"
            )

        await cl.Message(content=response_content).send()


@cl.on_chat_end
async def end():
    """Handle chat end"""
    await cl.Message(content="Thanks for using the Text2SQL Assistant! 👋").send()


if __name__ == "__main__":
    # Run with: chainlit run app.py
    pass
