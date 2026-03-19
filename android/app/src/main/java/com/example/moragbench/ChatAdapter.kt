package com.example.moragbench

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.*

class ChatAdapter(private val items: List<Message>) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

    companion object {
        private const val TYPE_USER = 1
        private const val TYPE_ASSISTANT = 2
    }

    override fun getItemViewType(position: Int): Int {
        return if (items[position].isUser) TYPE_USER else TYPE_ASSISTANT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return if (viewType == TYPE_USER) {
            val v = inflater.inflate(R.layout.item_user_message, parent, false)
            UserVH(v)
        } else {
            val v = inflater.inflate(R.layout.item_assistant_message, parent, false)
            AssistantVH(v)
        }
    }

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val msg = items[position]
        val timeStr = timeFormat.format(Date(msg.timeMillis))

        if (holder is UserVH) {
            holder.message.text = msg.content
            holder.time.text = timeStr
        } else if (holder is AssistantVH) {
            holder.message.text = msg.content
            holder.time.text = timeStr

            // Show metrics only after streaming is done and metrics exist
            val m = msg.metrics
            if (!msg.isStreaming && m != null) {
                // Format: "44.32 t/s  3.54 s"
                holder.metrics.visibility = View.VISIBLE
                holder.metrics.text = String.format(Locale.US, "%.2f t/s  %.2f s", m.tokensPerSec, m.durationSec)
            } else {
                holder.metrics.visibility = View.GONE
                holder.metrics.text = ""
            }
        }
    }

    class UserVH(v: View) : RecyclerView.ViewHolder(v) {
        val message: TextView = v.findViewById(R.id.messageText)
        val time: TextView = v.findViewById(R.id.messageTime)
    }

    class AssistantVH(v: View) : RecyclerView.ViewHolder(v) {
        val message: TextView = v.findViewById(R.id.messageText)
        val time: TextView = v.findViewById(R.id.messageTime)
        val metrics: TextView = v.findViewById(R.id.metricsText)
    }
}
