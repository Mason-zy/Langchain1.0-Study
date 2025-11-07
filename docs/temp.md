示例 6：RAG 问答 - 使用检索工具
======================================================================

问题: LangChain 有哪些核心组件？
回答: LangChain 的核心组件包括 Models、Prompts、Chains、Agents 和 Memory。这些 组件为构建和应用大型语言模型（LLM）提供了基础，能够支持诸如 RAG（检索增强生成）等高级功能。通过这些组件，开发者可以更好地利用 LLM 的能力，实现更智能和更高效的自然语言处理应用。
----------------------------------------------------------------------

问题: RAG 是什么？
回答: RAG 是 Retrieval-Augmented Generation 的缩写，它是一种结合了检索和生成的 技术，让语言模型能够访问外部知识库。它是 LangChain 框架中的一个核心应用场景，用于构建大型语言模型（LLM）应用。
----------------------------------------------------------------------

问题: LangChain 1.0 有什么改进？

错误: Error code: 400 - {'error': {'message': "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=search_knowledge_base {"query": "LangChain 1.0 \\u6539\\u9769"} </function>'}} 
Traceback (most recent call last):
  File "c:\Users\wangy\Desktop\temp\langchain_v1_study\phase2_practical\13_rag_basics\main.py", line 439, in main
    example_6_rag_qa(vectorstore)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "c:\Users\wangy\Desktop\temp\langchain_v1_study\phase2_practical\13_rag_basics\main.py", line 398, in example_6_rag_qa
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\pregel\main.py", line 3094, in invoke
    for chunk in self.stream(
                 ~~~~~~~~~~~^
        input,
        ^^^^^^
    ...<10 lines>...
        **kwargs,
        ^^^^^^^^^
    ):
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\pregel\main.py", line 2679, in stream
    for _ in runner.tick(
             ~~~~~~~~~~~^
        [t for t in loop.tasks.values() if not t.writes],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        schedule_task=loop.accept_push,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\pregel\_runner.py", line 167, in tick
    run_with_retry(
    ~~~~~~~~~~~~~~^
        t,
        ^^
    ...<10 lines>...
        },
        ^^
    )
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\pregel\_retry.py", line 42, in run_with_retry
    return task.proc.invoke(task.input, config)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 656, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 400, in invoke
    ret = self.func(*args, **kwargs)
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain\agents\factory.py", line 1065, in model_node
    response = _execute_model_sync(request)
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain\agents\factory.py", line 1038, in _execute_model_sync
    output = model_.invoke(messages)
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_core\runnables\base.py", line 5489, in invoke
    return self.bound.invoke(
           ~~~~~~~~~~~~~~~~~^
        input,
        ^^^^^^
        self._merge_configs(config),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        **{**self.kwargs, **kwargs},
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 382, in invoke
    self.generate_prompt(
    ~~~~~~~~~~~~~~~~~~~~^
        [self._convert_input(input)],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **kwargs,
        ^^^^^^^^^
    ).generations[0][0],
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1091, in generate_prompt   
    return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 906, in generate
    self._generate_with_cache(
    ~~~~~~~~~~~~~~~~~~~~~~~~~^
        m,
        ^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1195, in _generate_with_cache
    result = self._generate(
        messages, stop=stop, run_manager=run_manager, **kwargs
    )
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\langchain_groq\chat_models.py", line 544, in _generate
    response = self.client.create(messages=message_dicts, **params)
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\groq\resources\chat\completions.py", line 464, in create
    return self._post(
           ~~~~~~~~~~^
        "/openai/v1/chat/completions",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<45 lines>...
        stream_cls=Stream[ChatCompletionChunk],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\groq\_base_client.py", line 1242, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wangy\Desktop\temp\langchain_v1_study\venv\Lib\site-packages\groq\_base_client.py", line 1044, in request
    raise self._make_status_error_from_response(err.response) from None        
groq.BadRequestError: Error code: 400 - {'error': {'message': "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=search_knowledge_base {"query": "LangChain 1.0 \\u6539\\u9769"} </function>'}}
During task with name 'model' and id '54b7191c-5f00-8dbe-3594-6c5c2c884b06'  