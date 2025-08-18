import os
import re


def leggi_file() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "input.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste.")

    with open(file_path, "r") as file:
        return file.read().strip()


def pulisci_testo(testo: str) -> str:
    return re.sub(r"[^a-zA-Z\d\s:]", "", testo)


def main() -> None:
    testo = leggi_file()
    testo_pulito = pulisci_testo(testo)

    print("Contenuto originale di input.txt\n\n", testo)

    return


if __name__ == "__main__":
    main()
