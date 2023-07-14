"""Custom Chains
"""
from typing import Any, Dict, List, Tuple

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
    _collapse_docs,
    _split_list_of_docs,
)
from langchain.docstore.document import Document
from pydantic import root_validator


class MapReduceDocumentsChainV2(MapReduceDocumentsChain):
    """Custom MapReduce Document Chain. To allow better control of tokens at collapse and reduce step."""

    combine_max_tokens: int = 30000
    collapse_max_tokens: int = 5000

    @root_validator()
    def check_maximum_context_length(cls, values: Dict) -> Dict:
        max_token_dict = {
            "gpt-3.5-turbo": 3000,
            "gpt-3.5-turbo-16k": 14000,
            "gpt-4": 7000,
            "gpt-4-32k": 30000,
        }

        combine_doc_llm_model = values[
            "combine_document_chain"
        ].llm_chain.llm.model_name
        if combine_doc_llm_model in max_token_dict:
            if max_token_dict[combine_doc_llm_model] < values["combine_max_tokens"]:
                values["combine_max_tokens"] = max_token_dict[combine_doc_llm_model]

        if values["collapse_document_chain"]:
            collapse_doc_llm_model = values[
                "collapse_document_chain"
            ].llm_chain.llm.model_name
        else:
            collapse_doc_llm_model = values[
                "combine_document_chain"
            ].llm_chain.llm.model_name

        if collapse_doc_llm_model in max_token_dict:
            if max_token_dict[collapse_doc_llm_model] < values["collapse_max_tokens"]:
                values["collapse_max_tokens"] = max_token_dict[collapse_doc_llm_model]

        return values

    def combine_docs(
        self,
        docs: List[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        return self._process_results(results, docs, callbacks=callbacks, **kwargs)

    def _process_results(
        self,
        results: List[Dict],
        docs: List[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Process Result

        Args:
            results (List[Dict]): results
            docs (List[Document]): List of input documents to search over
            callbacks (Callbacks, optional): Callbacks. Defaults to None.

        Returns:
            Tuple[str, dict]: output, extra_return_dict
        """
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(results)
        ]
        length_func = self.combine_document_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
            """Function for collapsing sub groups of documents

            Args:
                docs (List[Document]): List of documents

            Returns:
                str: Summarized answer on small subgroups of documents
            """
            return self._collapse_chain.run(
                input_documents=docs, callbacks=callbacks, **kwargs
            )

        collapse_counter = 0
        while num_tokens is not None and num_tokens > self.combine_max_tokens:

            #
            collapse_counter += 1
            if collapse_counter == 2:
                raise Exception("Double Collapse steps. Stop")

            new_result_doc_list = _split_list_of_docs(
                result_docs, length_func, self.collapse_max_tokens, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = _collapse_docs(docs, _collapse_docs_func, **kwargs)
                result_docs.append(new_doc)
            num_tokens = self.combine_document_chain.prompt_length(
                result_docs, **kwargs
            )
        if self.return_intermediate_steps:
            _results = [r[self.llm_chain.output_key] for r in results]
            extra_return_dict = {"intermediate_steps": _results}
        else:
            extra_return_dict = {}
        output = self.combine_document_chain.run(
            input_documents=result_docs, callbacks=callbacks, **kwargs
        )
        return output, extra_return_dict
