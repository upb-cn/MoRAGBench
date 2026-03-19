package com.example.moragbench.ui

import android.app.Dialog
import android.content.Context
import android.view.*
import android.widget.TextView
import com.example.moragbench.R

class LoaderDialog(private val context: Context) {
    private var dialog: Dialog? = null

    /** Show the loader */
    fun show(message: String = "Loading…") {
        // Inflate layout
        val view = LayoutInflater.from(context).inflate(R.layout.dialog_loader, null)
        val msg = view.findViewById<TextView>(R.id.progressText)
        msg.text = message

        // Create a full-screen dialog
        dialog = Dialog(context, R.style.FullScreenLoaderTheme)
        dialog?.apply {
            setContentView(view)
            setCancelable(false)
            show()

            // Force full-screen layout
            window?.apply {
                setLayout(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                setBackgroundDrawableResource(android.R.color.transparent)
                setGravity(Gravity.CENTER)
            }
        }
    }

    /** Update message text */
    fun update(message: String) {
        dialog?.findViewById<TextView>(R.id.progressText)?.text = message
    }

    /** Hide loader */
    fun hide() {
        dialog?.dismiss()
        dialog = null
    }

    fun isShowing(): Boolean = dialog?.isShowing == true
}
