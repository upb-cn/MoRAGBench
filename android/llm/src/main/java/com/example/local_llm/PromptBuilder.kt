package com.example.local_llm

class PromptBuilder(
    private val tokenizer: TokenizerBridge,
    private val config: ModelConfig
) {
    fun buildPromptTokens(messages: List<Message>, intent: PromptIntent, systemPrompt: String, maxTokens: Int = 500): IntArray {
        return when (intent) {
            PromptIntent.CHAT -> buildQwenChatPrompt(messages, systemPrompt, maxTokens)
            PromptIntent.QA -> buildQwenQA(messages[0].text, systemPrompt)
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

    fun buildQwenChatPrompt(messages: List<Message>, systemPrompt: String, maxTokens: Int = 500): IntArray {
        val systemTokens = tokenizer.encode(systemPrompt)
        val assistantStart = config.roleTokenIds.assistantStart
        val end = config.roleTokenIds.endToken

        val conversationTokens = mutableListOf<Int>()
        conversationTokens.addAll(config.roleTokenIds.systemStart)
        conversationTokens.addAll(systemTokens.toList())
        conversationTokens.add(end)

        val turns = mutableListOf<Int>()
        for (msg in messages) {
            val roleTokens = if (msg.isUser) config.roleTokenIds.userStart else assistantStart
            val msgTokens = tokenizer.encode(msg.text)
            turns.addAll(roleTokens)
            turns.addAll(msgTokens.toList())
            turns.add(end)
        }

        val finalTurns = if (turns.size > maxTokens) {
            turns.takeLast(maxTokens)
        } else turns

        val result = mutableListOf<Int>()
        result.addAll(conversationTokens)
        result.addAll(finalTurns)
        result.addAll(assistantStart)

        return result.toIntArray()
    }
}
