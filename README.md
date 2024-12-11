# Market Research Agent
Multi agentic system that builds out an engine that will look up content and summarize it in a document

This agent takes the multi-hour process of market research and turns it into a 3 minute process.

## Set up
You'll also need wkhtmltopdf installed on your system for PDF generation. You can install it:
On Ubuntu/Debian:
```
sudo apt-get install -y wkhtmltopdf
```
On macOS:
```
brew install wkhtmltopdf
```

On Windows: Download the installer from [wkhtmltopdf downloads](https://wkhtmltopdf.org/downloads.html)

Create the python environment:
```sh
make create_environment
```

Install the dependencies:
```sh
make requirements
```

Copy the `.env.template` file to `.env` and add your OpenAI API key and Tavily API key.
```sh
cp template.env .env
```

If you already have these API keys defined in your .bashrc/.zshrc, add them to the `.env` file.
```sh
echo 'OPENAI_API_KEY="$OPENAI_API_KEY"' >> .env
echo 'TAVILY_API_KEY="$TAVILY_API_KEY"' >> .env
```

Or you can add them directly to the `.env` file:
```sh
echo 'OPENAI_API_KEY=<your-openai-api-key>' >> .env
echo 'TAVILY_API_KEY=<your-tavily-api-key>' >> .env
```




## Run the program

To run the program from CLI, use the following command:
```sh
python research_agent/main.py
```
This will run the default prompt defined in main.py.

You can also specify a custom reports directory:
```sh
python research_agent/main.py --reports-dir <path-to-reports-dir>
```



To run the Gradio interface, use the following command:
```sh
python research_agent/app.py
```