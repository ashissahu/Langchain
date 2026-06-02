from typing import Any, Dict, List
import streamlit as st
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Format the sources of the retrieved documents.

    Args:
        context_docs (List[Any]): _description_

    Returns:
        List[str]: _description_
    """
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None) #This line removes the "messages" key from the session state, effectively clearing the chat history. When the user clicks the "Clear chat" button, it triggers this action, and the chat interface will reset to its initial state, allowing for a fresh conversation without any previous messages displayed.
        st.rerun() #After clearing the messages, we call st.rerun() to immediately refresh the Streamlit app and reflect the changes in the UI. This ensures that the chat history is cleared and the user sees a clean slate for their next interaction.

#In below code, we check if "messages" is not already in the session state. If it's not, we initialize it with a default message from the assistant. 
# This message serves as a welcome prompt, inviting the user to ask questions about LangChain documentation. 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):#Here we are iterating through the list of messages stored in the session state and displaying each message in the chat interface. The st.chat_message function is used to create a chat bubble for each message, where the role (either "user" or "assistant") determines the styling of the bubble. The content of the message is displayed inside the bubble, and if there are any sources associated with the message, they are displayed in an expandable section below the message.
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")

#In Below code, when the user submits a prompt through the chat input, we append that prompt to the session state messages as a new entry with the role "user". 
# We then display the user's message in the chat interface by calling st.markdown(prompt).
#  Next, we create a new chat message for the assistant's response and use a spinner to indicate that the system is processing the request.
# We call the run_llm function to get the answer and context based on the user's query . 
# Once we receive the result, we extract the answer and sources, display the answer in the chat interface by calling st.markdown(answer), 
# and if there are sources, we display them in an expandable section. 
# Finally, we append the assistant's response to the session state messages, including both the answer and the sources, so that it is stored in the chat history for future reference.

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)