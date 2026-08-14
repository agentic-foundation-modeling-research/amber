"""base class for evaluation"""
# answer string match
import collections
import html
import importlib
import json
import re
import time
import urllib
from pathlib import Path
from typing import Any, Tuple, Union

from beartype import beartype
from nltk.tokenize import word_tokenize  # type: ignore
from playwright.sync_api import CDPSession, Page
import openai

from .helper_functions import (
    PseudoPage,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_post_url,
    reddit_normalize_url,
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)


_COORDINATE_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_COORDINATE_MARKER_RE = re.compile(
    rf"\bcoordinates?\s*:\s*({_COORDINATE_NUMBER_RE.pattern})\s*,\s*({_COORDINATE_NUMBER_RE.pattern})",
    re.IGNORECASE,
)


def _is_valid_coordinate_pair(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def extract_coordinate_pair(answer: str) -> tuple[float, float] | None:
    marker_match = _COORDINATE_MARKER_RE.search(answer)
    if marker_match:
        lat = float(marker_match.group(1))
        lon = float(marker_match.group(2))
        if _is_valid_coordinate_pair(lat, lon):
            return lat, lon
        return None

    numbers = [float(match.group(0)) for match in _COORDINATE_NUMBER_RE.finditer(answer)]
    for lat, lon in zip(numbers, numbers[1:]):
        if _is_valid_coordinate_pair(lat, lon):
            return lat, lon
    return None


class Evaluator(object):
    def __init__(self, configs, eval_tag: str = "", **kwargs) -> None:
        self.configs = configs
        self.eval_tag = eval_tag
        self.client = openai.OpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"])

    @beartype
    def __call__(
        self,
        trajectory: list[dict], # fake [{, last_action}]
        config_file: Path | str,
        page: Page | PseudoPage,
        client: CDPSession,
    ) -> float:
        raise NotImplementedError


class StringEvaluator(Evaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @staticmethod
    @beartype
    def ref_or_values(ref: str) -> list[str]:
        return ref.split(" |OR| ")

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: str) -> float:
        clean_pred = StringEvaluator.clean_answer(pred)
        return float(
            any(
                clean_pred == StringEvaluator.clean_answer(ref_value)
                for ref_value in StringEvaluator.ref_or_values(ref)
            )
        )

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_pred = StringEvaluator.clean_answer(pred)
        for ref_value in StringEvaluator.ref_or_values(ref):
            clean_ref = StringEvaluator.clean_answer(ref_value)
            # tokenize the answer if the ref is a single word
            # prevent false positive (e.g, 0)
            if (
                tokenize
                and len(clean_ref) == 1
                and len(word_tokenize(clean_ref)) == 1
            ):
                tok_pred = word_tokenize(clean_pred)
                if clean_ref in tok_pred:
                    return 1.0
            elif clean_ref in clean_pred:
                return 1.0
        return 0.0

    @staticmethod
    @beartype
    def fuzzy_match(client, ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(client, pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(client, ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(client, pred, ref, intent)

    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage | None = None,
        client: CDPSession | None = None,
    ) -> float:
        
        configs = self.configs

        answer = trajectory[1]["answer"]
        pred = self.clean_answer(answer)

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                self.client,
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                self.client, ref=reference, pred=pred, intent=intent
                            )
        return score


class CoordinateEvaluator(Evaluator):
    """Check whether an answer contains latitude/longitude within tolerance."""

    @beartype
    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage | None = None,
        client: CDPSession | None = None,
    ) -> float:
        configs = self.configs
        answer = trajectory[1]["answer"]
        pred_coords = extract_coordinate_pair(answer)
        if pred_coords is None:
            return 0.0

        reference_answers = configs["eval"]["reference_answers"]
        ref_coords = reference_answers["coordinate_match"]
        assert isinstance(ref_coords, list)
        assert len(ref_coords) == 2

        ref_lat = float(ref_coords[0])
        ref_lon = float(ref_coords[1])
        pred_lat, pred_lon = pred_coords
        tolerance = float(configs["eval"].get("coordinate_tolerance", 0.01))

        return float(
            abs(pred_lat - ref_lat) <= tolerance
            and abs(pred_lon - ref_lon) <= tolerance
        )


class URLEvaluator(Evaluator):
    """Check URL matching"""

    @beartype
    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        
        configs = self.configs

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        def resolve_url(url: str) -> str:
            """Resolve helper function URL references."""
            if url.startswith("func:"):
                func = url.split("func:", 1)[1]
                func = func.replace("__last_url__", page.url)
                url = eval(func)
            return url

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(resolve_url(url)) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "GOLD in PRED")
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            score = base_score * query_score

        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return score
    

class ApproxURLEvaluator(Evaluator):

    @beartype
    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        
        configs = self.configs

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        def resolve_url(url: str) -> str:
            """Resolve helper function URL references."""
            if url.startswith("func:"):
                func = url.split("func:", 1)[1]
                func = func.replace("__last_url__", page.url)
                url = eval(func)
            return url

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(resolve_url(url)) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "GOLD in PRED")
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            base_score = float(
                any(
                    [
                        pred_base_paths.startswith(ref_base_path)
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            # TODO: check if we need this
            # for k, possible_values in ref_queries.items():
            #     query_score *= float(
            #         any(
            #             possible_ref_value in pred_query.get(k, [])
            #             for possible_ref_value in possible_values
            #         )
            #     )
            score = base_score * query_score

        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return score


class HTMLContentEvaluator(Evaluator):
    """Check whether the contents appear in the page"""

    @beartype
    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage,
        client: CDPSession | None = None,
    ) -> float:
        
        configs = self.configs

        targets = configs["eval"]["program_html"]

        score = 1.0
        for target in targets:
            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)
                print(target_url)

            locator: str = target["locator"]  # js element locator

            # navigate to that url
            prev_page = None
            if target_url != "last":
                prev_page = page
                page = page.context.new_page()
                page.goto(target_url)
                time.sleep(3)  # TODO [shuyanzh]: fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                selected_element = page.content()
            # use JS to select the element
            elif locator.startswith("document.") or locator.startswith(
                "[...document."
            ):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    # the page is wrong, return empty
                    selected_element = ""
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = eval(func)
            else:
                raise ValueError(f"Unknown locator: {locator}")

            selected_element = html.unescape(selected_element)

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                cur_score = StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
                score *= float(cur_score)
                # print(f"[exact match] {cur_score}, selected element: {selected_element}, required contents: {required_contents}")
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any(
                        [
                            StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
                    score *= float(cur_score)
                    # print(f"[must include] {cur_score}, selected element: {selected_element}, required contents: {content_or}")
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )

            if prev_page:
                page.close()
                page = prev_page
                prev_page = None

        return score


class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
        trajectory: list[dict],
        page: Page | PseudoPage,
        client: CDPSession | None,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(trajectory, page, client)
            score *= cur_score
        return score


@beartype
def evaluator_router(configs: dict, api_key=None, base_url=None) -> EvaluatorComb:
    
    """Router to get the evaluator class"""

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(configs, api_key=api_key, base_url=base_url))
            case "coordinate_match":
                evaluators.append(CoordinateEvaluator(configs, api_key=api_key, base_url=base_url))
            case "url_match":
                evaluators.append(URLEvaluator(configs, api_key=api_key, base_url=base_url))
            case "approx_url_match":
                evaluators.append(ApproxURLEvaluator(configs, api_key=api_key, base_url=base_url))
            case "program_html":
                evaluators.append(HTMLContentEvaluator(configs, api_key=api_key, base_url=base_url))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)
