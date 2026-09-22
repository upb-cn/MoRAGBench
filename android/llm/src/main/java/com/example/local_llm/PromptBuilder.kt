package com.example.local_llm

class PromptBuilder(
    private val tokenizer: TokenizerBridge,
    private val config: ModelConfig
) {
    fun buildPromptTokens(messages: List<Message>, intent: PromptIntent, systemPrompt: String, modelFamily: String, maxPromptTokens: Int = DEFAULT_MAX_PROMPT_TOKENS): IntArray {
        val isLlama = modelFamily == "llama"
        return when (intent) {
            PromptIntent.CHAT -> buildQwenChatPrompt(messages, systemPrompt, maxPromptTokens)
            PromptIntent.QA -> if (isLlama) buildLlamaQA(messages[0].text, systemPrompt) else buildQwenQA(messages[0].text, systemPrompt)
        }
    }

    private fun buildQwenQA(userInput: String, systemPrompt: String): IntArray {
        val systemPrompt = systemPrompt
        val userPrompt = "Question: $userInput\nAnswer:"

        val systemTokens = tokenizer.encode(systemPrompt)
        val userTokens = tokenizer.encode(userPrompt)

        return buildList {
            addAll(config.roleTokenIds.systemStart)
            addAll(systemTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.userStart)
            addAll(userTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.assistantStart)
        }.toIntArray()
    }
    private fun buildLlamaQA(userInput: String, systemPrompt: String): IntArray {
        val userPrompt = "Question: $userInput\nAnswer:"

        val systemTokens = tokenizer.encode(systemPrompt)
        val userTokens = tokenizer.encode(userPrompt)
        val newlineTokens = tokenizer.encode("\n\n").toList()
        val bosToken = tokenizer.getTokenId("<|begin_of_text|>")

        val result = buildList {
            add(bosToken)

            addAll(config.roleTokenIds.systemStart)
            addAll(newlineTokens)
            addAll(systemTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.userStart)
            addAll(newlineTokens)
            addAll(userTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.assistantStart)
            addAll(newlineTokens)
        }.toIntArray()
        return result
    }

    /**
     * Builds the ChatML prompt, bounded by [maxPromptTokens].
     *
     * That budget is deliberately separate from the generation budget. They used to
     * be the same value, which meant a 50-token generation limit also capped the
     * prompt at 50 tokens and silently cut the front off most questions.
     */
    fun buildQwenChatPrompt(messages: List<Message>, systemPrompt: String, maxPromptTokens: Int = DEFAULT_MAX_PROMPT_TOKENS): IntArray {
        val systemTokens = tokenizer.encode(systemPrompt)
        val assistantStart = config.roleTokenIds.assistantStart
        val end = config.roleTokenIds.endToken

        val conversationTokens = mutableListOf<Int>()
        conversationTokens.addAll(config.roleTokenIds.systemStart)
        conversationTokens.addAll(systemTokens.toList())
        conversationTokens.add(end)

        // What the conversation turns may occupy, once the system turn and the
        // trailing assistant marker are paid for.
        var remaining = maxPromptTokens - conversationTokens.size - assistantStart.size

        // Newest turn first, so that when the budget runs out it is the oldest turns
        // that go. Within a turn the head is kept rather than the tail, so a question
        // that precedes its retrieved documents survives and the documents are cut.
        // The role markers are never dropped, which the old tail-truncation did.
        val turnGroups = mutableListOf<List<Int>>()
        for (msg in messages.asReversed()) {
            val roleTokens = if (msg.isUser) config.roleTokenIds.userStart else assistantStart
            val overhead = roleTokens.size + 1 // role marker plus the end token
            if (remaining <= overhead) break

            val body = tokenizer.encode(msg.text).toList()
            val bodyBudget = remaining - overhead
            val kept = if (body.size > bodyBudget) body.take(bodyBudget) else body

            turnGroups.add(roleTokens + kept + end)
            remaining -= overhead + kept.size
        }
        turnGroups.reverse()

        val result = mutableListOf<Int>()
        result.addAll(conversationTokens)
        turnGroups.forEach { result.addAll(it) }
        result.addAll(assistantStart)

        return result.toIntArray()
    }

    companion object {
        /** Used when no budget is supplied; benchmark runs always pass one explicitly. */
        const val DEFAULT_MAX_PROMPT_TOKENS = 2048
    }
}
