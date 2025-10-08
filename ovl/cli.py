import click

from ovl.app import app


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to run the server on")
@click.option("--port", default=7860, help="Port to run the server on")
@click.option("--share", is_flag=True, help="Create a public link")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(host, port, share, debug):
    """CLI to launch Gradio."""
    app.launch(server_name=host, server_port=port, share=share, debug=debug)


if __name__ == "__main__":
    main()
