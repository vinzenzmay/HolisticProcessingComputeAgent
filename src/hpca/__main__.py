"""Entry point: ``hpca`` console script / ``python -m hpca``."""

from hpca.tui.app import HpcaApp


def main() -> None:
    HpcaApp().run()


if __name__ == "__main__":
    main()
