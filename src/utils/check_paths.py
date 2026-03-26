from pathlib import Path

def verify_setup():
    # 1. Define a raiz do projeto (caminho onde o script está - 2 níveis acima)
    # Ex: MathSolverAI/src/utils/check_paths.py -> MathSolverAI/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    print(f"--- Diagnóstico do MathSolverAI ---")
    print(f"Diretório Raiz: {BASE_DIR}")
    
    # 2. Define os caminhos das pastas de dados
    data_raw = BASE_DIR / "data" / "raw"
    data_proc = BASE_DIR / "data" / "processed"
    
    # 3. Verifica se as pastas existem
    folders = [data_raw, data_proc]
    for folder in folders:
        if folder.exists():
            print(f"[OK] Pasta encontrada: {folder.relative_to(BASE_DIR)}")
        else:
            print(f"[ERRO] Pasta NÃO encontrada: {folder}")

    # 4. Lista arquivos na pasta 'raw'
    print(f"\n--- Arquivos em 'data/raw' ---")
    files = list(data_raw.glob("*")) # Pega tudo que estiver lá dentro
    
    if not files:
        print("(!) A pasta está vazia. Coloque uma imagem de teste aqui!")
    else:
        for file in files:
            # Verifica se é uma extensão de imagem comum
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"- [IMAGEM] {file.name}")
            else:
                print(f"- [OUTRO]  {file.name}")

if __name__ == "__main__":
    verify_setup()