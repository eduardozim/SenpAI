#!/bin/bash

# ==========================================
# Script de Instalação do SenpAI para Linux
# ==========================================

echo "Iniciando a instalação do SenpAI no Linux..."

# 1. Verifica se apt-get está disponível para dependências do sistema (Debian/Ubuntu)
if command -v apt-get &> /dev/null; then
    echo "Instalando pacotes do sistema necessários (Python 3.11, venv, libgl1 para OpenCV)..."
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential libgl1-mesa-glx libglib2.0-0
else
    echo "Aviso: 'apt-get' não encontrado. Certifique-se de que Python 3.11 e as dependências do OpenCV estejam instaladas no seu sistema."
fi

# 2. Criar ambiente virtual
echo "Criando o ambiente virtual (.venv)..."
if command -v python3.11 &> /dev/null; then
    python3.11 -m venv .venv
else
    python3 -m venv .venv
fi

# 3. Ativar o ambiente virtual e instalar as dependências
echo "Ativando o ambiente virtual e instalando as bibliotecas..."
source .venv/bin/activate

# Atualiza pip
pip install --upgrade pip

# Instala requerimentos base
pip install -r requirements.txt

# (Opcional) Tenta instalar dependências CUDA se houver GPU NVIDIA e nvcc instalado
if command -v nvidia-smi &> /dev/null; then
    echo "Placa NVIDIA detectada. Instalando aceleração CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install ultralytics
fi

echo "=========================================="
echo "Instalação concluída com sucesso!"
echo "=========================================="
echo ""
echo "Para iniciar o SenpAI, execute os comandos abaixo:"
echo ""
echo "  source .venv/bin/activate"
echo "  python -m streamlit run app.py"
echo ""
