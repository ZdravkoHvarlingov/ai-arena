from .render import Render


def main():
    with Render(800, 600) as r:
        r.loop()


if __name__ == '__main__':
    main()
