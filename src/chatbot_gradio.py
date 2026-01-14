"""
Medical Knowledge Graph Chatbot with Gradio
============================================
Interactive chatbot interface for medical question answering
using the improved inference pipeline.

Author: CVDGraphRAG Team
"""

import os
import gradio as gr
from camel.storages import Neo4jGraph
from inference_utils import infer
from logger_ import get_logger

logger = get_logger("chatbot_gradio", log_file="logs/chatbot_gradio.log")

def initialize_neo4j():
    """Initialize Neo4j graph database connection"""
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([url, username, password]):
        logger.error("Missing Neo4j credentials in environment variables")
        raise ValueError("Please set NEO4J_URL, NEO4J_USERNAME, and NEO4J_PASSWORD")
    
    logger.info(f"Connecting to Neo4j at {url}")
    n4j = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )
    logger.info("Neo4j connection established")
    return n4j

n4j = None

def chat_inference(message, history, use_multi_subgraph):
    """
    Process user question and return answer
    
    Args:
        message: User's question
        history: Chat history (not used in current implementation)
        use_multi_subgraph: Whether to use multi-subgraph mode
    
    Returns:
        Answer string
    """
    global n4j
    
    if n4j is None:
        return "Error: Neo4j connection not initialized. Please restart the application."
    
    if not message or not message.strip():
        return "Please enter a question."
    
    logger.info(f"Processing question: {message[:100]}...")
    logger.info(f"Multi-subgraph mode: {use_multi_subgraph}")
    
    try:
        answer = infer(n4j, message, use_multi_subgraph=use_multi_subgraph)
        
        if answer:
            logger.info(f"Answer generated successfully ({len(answer)} chars)")
            return answer
        else:
            logger.warning("No answer generated")
            return "Could not generate an answer. Please try rephrasing your question or ensure the knowledge graph contains relevant data."
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return f"Error: {str(e)}\n\nPlease check the logs for more details."

def check_database_status():
    """Check if the database has data"""
    global n4j
    
    if n4j is None:
        return "Not connected to Neo4j"
    
    try:
        query = """
        MATCH (s:Summary)
        RETURN count(s) as summary_count
        """
        result = n4j.query(query)
        
        if result and result[0]['summary_count'] > 0:
            count = result[0]['summary_count']
            return f"Connected | {count} summaries in database"
        else:
            return "Connected but database is empty. Please build the graph first."
    
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return f"Error: {str(e)}"

def create_interface():
    """Create Gradio ChatInterface"""
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Heart/Cardiovascular Failure Medical Knowledge Graph Chatbot") as demo:
        gr.Markdown(
            """
            # Heart/Cardiovascular Failure Medical Knowledge Graph Chatbot
            
            Ask questions multiple choices about heart/cardiovascular failure medical reports and conditions.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    show_label=True,
                    avatar_images=(None, "/home/medgraph/src/images.jpg")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Enter your medical question here...",
                        lines=2,
                        scale=4
                    )
                    # submit_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    multi_subgraph = gr.Checkbox(
                        label="CDVGraphRAG Multi-Subgraph Mode",
                        value=False
                    )
        
        def respond(message, chat_history, use_multi):
            if not message.strip():
                return chat_history, ""
            
            chat_history.append((message, None))
            
            bot_response = chat_inference(message, chat_history, use_multi)
            
            chat_history[-1] = (message, bot_response)
            
            return chat_history, ""
        
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, multi_subgraph],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, multi_subgraph],
            outputs=[chatbot, msg]
        )
        
        # Clear button
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        # Refresh status button
        # refresh_btn.click(
        #     check_database_status,
        #     inputs=None,
        #     outputs=status_output
        # )
        
        # Examples
        # gr.Examples(
        #     examples=[
        #         "What are the main symptoms of the patient?",
        #         "What treatments were recommended?",
        #         "What is the diagnosis for this patient?",
        #         "Are there any complications mentioned?",
        #         "What medications were prescribed?"
        #     ],
        #     inputs=msg,
        #     label="Example Questions"
        # )
    
    return demo

def main():
    global n4j
    
    logger.info("Starting Medical Knowledge Graph Chatbot")
    
    try:
        n4j = initialize_neo4j()
        logger.info("Neo4j initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {str(e)}", exc_info=True)
        print(f"Error: Could not connect to Neo4j - {str(e)}")
        print("\nPlease ensure:")
        print("1. Neo4j is running")
        print("2. Environment variables are set (NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)")
        return
    
    demo = create_interface()
    
    logger.info("Launching Gradio interface...")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=True,              # Create public gradio.live link
        show_error=True,
        show_api=False
    )

if __name__ == "__main__":
    main()
