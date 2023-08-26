"""Custom Chains
"""
from typing import Any, Dict, List, Tuple, Optional, Callable

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.reduce import _collapse_docs, _acollapse_docs
from langchain.docstore.document import Document
from pydantic import root_validator

def _split_list_of_docs(
    docs: List[Document], length_func: Callable, collapse_token_max: int, **kwargs: Any
) -> List[List[Document]]:
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > collapse_token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError(
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list

class ReduceDocumentsChainV2(ReduceDocumentsChain):
    combine_max_tokens: Optional[int] = None
    collapse_max_tokens: Optional[int] = None
    
    @root_validator()
    def check_maximum_context_length(cls, values: Dict) -> Dict:
        values["combine_max_tokens"] = values.get("combine_max_tokens") or values["token_max"]
        values["collapse_max_tokens"] = values.get("collapse_max_tokens") or values["token_max"]
                
        max_token_dict = {
            "gpt-3.5-turbo": 3000,
            "gpt-3.5-turbo-16k": 14000,
            "gpt-4": 7000,
            "gpt-4-32k": 30000,
        }

        combine_doc_llm_model = values[
            "combine_documents_chain"
        ].llm_chain.llm.model_name
        if combine_doc_llm_model in max_token_dict:
            if max_token_dict[combine_doc_llm_model] < values["combine_max_tokens"]:
                values["combine_max_tokens"] = max_token_dict[combine_doc_llm_model]

        if values["collapse_documents_chain"]:
            collapse_doc_llm_model = values[
                "collapse_documents_chain"
            ].llm_chain.llm.model_name
        else:
            collapse_doc_llm_model = values[
                "combine_documents_chain"
            ].llm_chain.llm.model_name

        if collapse_doc_llm_model in max_token_dict:
            if max_token_dict[collapse_doc_llm_model] < values["collapse_max_tokens"]:
                values["collapse_max_tokens"] = max_token_dict[collapse_doc_llm_model]
        
        return values

    def combine_docs(
        self,
        docs: List[Document],
        combine_token_max: Optional[int] = None,
        collapse_token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine multiple documents recursively.

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        result_docs, _ = self._collapse(
            docs=docs,
            combine_token_max=combine_token_max,
            collapse_token_max=collapse_token_max,
            callbacks=callbacks,
            **kwargs
        )
        return self.combine_documents_chain.combine_docs(
            docs=result_docs, callbacks=callbacks, **kwargs
        )

    async def acombine_docs(
        self,
        docs: List[Document],
        combine_token_max: Optional[int] = None,
        collapse_token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Async combine multiple documents recursively.

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        result_docs, _ = await self._acollapse(
            docs, combine_token_max=combine_token_max, collapse_token_max=collapse_token_max, callbacks=callbacks, **kwargs
        )
        return await self.combine_documents_chain.acombine_docs(
            docs=result_docs, callbacks=callbacks, **kwargs
        )

    def _collapse(
        self,
        docs: List[Document],
        combine_token_max: Optional[int] = None,
        collapse_token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[List[Document], dict]:
        result_docs = docs
        length_func = self.combine_documents_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
            return self._collapse_chain.run(
                input_documents=docs, callbacks=callbacks, **kwargs
            )

        _combine_token_max = combine_token_max or self.combine_max_tokens
        _collapse_token_max = collapse_token_max or self.collapse_max_tokens
        collapse_counter = 0
        while num_tokens is not None and num_tokens > _combine_token_max:
            collapse_counter += 1
            if collapse_counter > 1:
                raise Exception("Document Collapse more than once!!!")
                
            new_result_doc_list = _split_list_of_docs(
                docs=result_docs,
                length_func=length_func,
                collapse_token_max=_collapse_token_max,
                **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = _collapse_docs(docs, _collapse_docs_func, **kwargs)
                result_docs.append(new_doc)
            num_tokens = length_func(result_docs, **kwargs)
        return result_docs, {}

    async def _acollapse(
        self,
        docs: List[Document],
        combine_token_max: Optional[int] = None,
        collapse_token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[List[Document], dict]:
        result_docs = docs
        length_func = self.combine_documents_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        async def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
            return await self._collapse_chain.arun(
                input_documents=docs, callbacks=callbacks, **kwargs
            )

        _combine_token_max = combine_token_max or self.combine_max_tokens
        _collapse_token_max = collapse_token_max or self.collapse_max_tokens
        while num_tokens is not None and num_tokens > _combine_token_max:
            new_result_doc_list = _split_list_of_docs(
                result_docs, length_func, _collapse_token_max, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = await _acollapse_docs(docs, _collapse_docs_func, **kwargs)
                result_docs.append(new_doc)
            num_tokens = length_func(result_docs, **kwargs)
        return result_docs, {}

    @property
    def _chain_type(self) -> str:
        return "reduce_documents_chain"