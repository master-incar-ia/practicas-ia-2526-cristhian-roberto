class Brocha:
    def __init__(self, color: str, tamannio: int):
        self.tamannio = tamannio
        self.color = color

    def pintar(self) -> str:
        return f"Brocha de tamaño {self.tamannio}, con color {self.color}."


if __name__ == "__main__":
    brocha_azul = Brocha("azul", 5)
    brocha_verde_gigante = Brocha("verde", 10)
    brocha_azul_pequeña = Brocha("azul", 2)

    brocha_azul.pintar()
