""" 
TRABALHO DE INTELIGÊNCIA ARTIFICIAL

Equipe:
    - Agnaldo Erick Maia de Oliveira (539650) [ES]
    - Francisco Rodrigo de Santiago Pinheiro (554394) [ES]
    - Vitor Costa de Sousa (536678) [ES]
"""


def app() -> None:
    print("Hello World!")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print('Errors:', e.args)
