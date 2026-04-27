from typing import Any, Dict, List, Optional


class PromptGenerator:
    """Prompt formatter for retrieval-augmented QA."""

    @staticmethod
    def _format_kormed_question_with_options(question: str, options: Optional[Dict[str, str]]) -> str:
        if not options:
            return question

        option_lines: List[str] = []
        for idx, key in enumerate(["A", "B", "C", "D", "E"], start=1):
            text = (options.get(key) or "").strip()
            if not text:
                continue
            option_lines.append(f"{idx}) {text}")

        if not option_lines:
            return question

        return f"{question}\n" + "\n".join(option_lines)

    @staticmethod
    def _build_reasoning_template(dataset: Optional[str], q_type: Optional[int]) -> str:
        is_kormed = (dataset or "").lower() == "kormedmcqa"
        is_mcq = is_kormed or q_type == 1

        if is_mcq:
            if is_kormed:
                answer_line = "답: [1~5 숫자 중 하나]"
            else:
                answer_line = "답: 1) [정답]"
            return (
                "- 문제 분석: ...\n"
                "- 가능성 있는 답안 분석: ...\n"
                "- 정답 결정 이유: ...\n"
                f"{answer_line}"
            )

        return (
            "- 문제 분석: ...\n"
            "- 정답 도출 과정: ...\n"
            "답: [정답]"
        )

    @staticmethod
    def _build_prompt(
        docs: Optional[List[str]],
        question: str,
        seq_type: str,
        dataset: Optional[str],
        q_type: Optional[int],
        options: Optional[Dict[str, str]],
    ) -> str:
        is_kormed = (dataset or "").lower() == "kormedmcqa"
        question_text = (
            PromptGenerator._format_kormed_question_with_options(question, options)
            if is_kormed
            else question
        )
        reasoning_template = PromptGenerator._build_reasoning_template(dataset=dataset, q_type=q_type)

        prompt = (
            "당신은 전문 의학 지식을 가진 의사입니다. 주어진 문제에 대해 논리적인 사고 과정을 통해 정답을 도출하세요. "
            "각 단계를 명확히 나누어 작성하고 마지막에 정답을 제시하세요.\n"
            "Format:\n"
            f"{reasoning_template}\n\n"
        )

        if docs is not None:
            prompt += "반드시 아래 참고 문맥의 내용만 근거로 답변하세요. 문맥에 근거가 없으면 모른다고 답하세요.\n"

        if seq_type == "qd":
            prompt += f"\n문제: {question_text}\n"

        if docs is not None:
            for idx, doc in enumerate(docs, start=1):
                prompt += f"\n참고 문맥 {idx}: {doc}\n"

        if seq_type == "dq":
            prompt += f"\n문제: {question_text}\n"

        prompt += "\n답변:"
        return prompt

    @staticmethod
    def generate_answer_with_docs(
        docs: List[str],
        question: str,
        seq_type: str = "dq",
        dataset: Optional[str] = None,
        q_type: Optional[int] = None,
        options: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        meta = metadata or {}
        prompt = PromptGenerator._build_prompt(
            docs=docs,
            question=question,
            seq_type=seq_type,
            dataset=dataset or meta.get("dataset"),
            q_type=q_type if q_type is not None else meta.get("q_type"),
            options=options or meta.get("options"),
        )
        return [prompt]

    @staticmethod
    def generate_answer_without_docs(
        question: str,
        dataset: Optional[str] = None,
        q_type: Optional[int] = None,
        options: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        meta = metadata or {}
        prompt = PromptGenerator._build_prompt(
            docs=None,
            question=question,
            seq_type="dq",
            dataset=dataset or meta.get("dataset"),
            q_type=q_type if q_type is not None else meta.get("q_type"),
            options=options or meta.get("options"),
        )
        return [prompt]
