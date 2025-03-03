import streamlit as st
import os
import tempfile
from processing import FactoryLoader

# Configuração inicial
st.set_page_config(page_title="Chat com Documentos", layout="wide")

# Funções de processamento
def process_input(uploaded_files):
    """Processa todos os inputs e retorna texto consolidado"""
    full_text = ""

    
    # Processar arquivos
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            loader = FactoryLoader()
            processor = loader.get_loader(file.type,file)
    
    return processor

# Interface principal
def main():
    st.title("📚 Chat Inteligente com Documentos")
    
    # Sidebar para upload
    with st.sidebar:
        st.header("Carregar Documentos")
        uploaded_files = st.file_uploader(
            "Arraste arquivos PDF/TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Processar Documentos"):
            with st.spinner("Processando..."):
                st.session_state.full_text = process_input(uploaded_files)
                st.success("Documentos prontos para consulta!")
                

    # Área de chat
    if "full_text" not in st.session_state:
        st.info("Carregue documentos para começar")
        return

    # Histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta sobre os documentos"):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gerar resposta
        with st.spinner("Pensando..."):
            try:
                # Substituir por sua integração com LangChain
                ia = process_input(uploaded_files)
                response = ia.get_relevant_documents(prompt)
                resposta = ""
                for r in response:
                    resposta += r.page_content
            except Exception as e:
                response = f"Erro: {str(e)}"

        # Adicionar e exibir resposta
        st.session_state.messages.append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)

if __name__ == "__main__":
    main()