import os
import re


def leggi_file() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "input.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste.")

    with open(file_path, "r") as file:
        return file.read().strip()


def pulisci_testo(testo: str) -> str:
    testo = testo.lower()
    testo = re.sub(r"[^a-zA-Z\d\s:]", "", testo)

    return testo


def conta_righe(testo: str) -> int:
    return len(testo.splitlines())


def conta_parole(testo: str) -> dict[str, int]:
    parole: dict[str, int] = {}

    for parola in testo.split():
        parola = parola.strip()

        if parola:
            parole[parola] = parole.get(parola, 0) + 1

    return parole


def top_5_parole(parole: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(parole.items(), key=lambda x: x[1], reverse=True)[:5]


def genera_report(parole: dict[str, int]) -> str:
    report = "Top 5 parole più frequenti:\n"
    for parola, conteggio in top_5_parole(parole):
        report += f"{parola}: {conteggio}\n"
    return report


def main() -> None:
    testo = leggi_file()
    testo_pulito = pulisci_testo(testo)

    print("Contenuto originale di input.txt\n\n", testo)

    numero_righe = conta_righe(testo_pulito)
    parole = conta_parole(testo_pulito)

    print(f"\nNumero di righe: {numero_righe}")
    print(f"\nNumero di parole totale: {sum(parole.values())}")

    report = genera_report(parole)
    print(f"\n{report}")

    return


if __name__ == "__main__":
    main()
