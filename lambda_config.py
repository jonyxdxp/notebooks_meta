{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "955bd9ee",
   "metadata": {},
   "source": [
    "1. Crear la instancia\n",
    "\n",
    "Entra a lambdalabs.com → Cloud → Launch Instance\n",
    "Elige A10 (24GB) — suficiente para tus modelos y más barato que A100\n",
    "Región: us-east-1 o us-west-2 (más disponibilidad)\n",
    "Agrega tu SSH key antes de lanzar (Settings → SSH Keys)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0192592c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. conectarte\n",
    "\n",
    "ssh ubuntu@<ip-address>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "827e577b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Setup del entorno (una sola vez)\n",
    "\n",
    "# Lambda ya viene con CUDA, PyTorch y conda preinstalados — verificar versiones\n",
    "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n",
    "\n",
    "# Clonar tu repo\n",
    "git clone https://github.com/jonyxdxp/notebooks_meta.git\n",
    "cd notebooks_meta\n",
    "\n",
    "# Instalar dependencias\n",
    "pip install -r requirements.txt"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
