"""
TDS Project - LLM based Assistant
"""

import os
import subprocess
import sqlite3
import json
from datetime import datetime
import requests
from flask import Flask, request, jsonify
from PIL import Image

# Retrieve environment variables
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai"

print(AIPROXY_TOKEN)

# Set up the headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

app = Flask(__name__)

function_descriptions = [
    {
        "name": "run_datagen",
        "description": "Install UV if not installed already, then run the datagen.py from the path hardcoded in the script",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "format_file",
        "description": "Format a Markdown file using Prettier",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "count_number_of_days",
        "description": "Count the number of given days in a list of dates",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "sort_contacts",
        "description": "Sort contacts by last name and then by first name",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }    
    },
    {
        "name": "extract_log_lines",
        "description": "Extract the first line from the 10 most recent log files",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "generate_markdown_index",
        "description": "Generate an index of Markdown files and their first headings",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "extract_email_sender",
        "description": "Extract the sender's email address from an email content using GPT model",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "extract_credit_card_number",
        "description": "Extract credit card number from an image using GPT model",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "find_similar_comments",
        "description": "Find the most similar pair of comments using GPT model",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "calculate_ticket_sales", 
        "description": "Calculate total sales for Gold tickets from a SQLite database",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "fetch_data_from_api",
        "description": "Fetch data from an API and save it as JSON",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                }
            },
            "required": ["apiEndPoint"]
        }
    },
    {
        "name": "clone_git_repo",
        "description": "Clone a Git repository to a specific directory",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                }
            },
            "required": ["repoLink"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Execute a SQL query to calculate total sales for Gold tickets",
        "parameters": [{
            "type": "object",
            "properties": {
                "dbLocation": {
                    "type": "string",
                    "description": "The location and name of the database to be read"
                },
                "fileLocation": {
                    "type": "string",
                    "description": "The location and name of the new file on which the result will be written"
                },
                "sqlQuery": {
                    "type": "string",
                    "description": "This is the SQL query to be executed on the database"
                }
            },
            "required": ["dbLocation","fileLocation","sqlQuery"]
        },
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                }
            },
            "required": ["task"]
        },
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                }
            },
            "required": ["task"]
        }]
    },
    {
        "name": "scrape_website",
        "description": "Scrape a website and save its content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                }
            },
            "required": ["websiteLink"]
        }
    },
    {
        "name": "process_image",
        "description": "Resize an image to 50% of its original size",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe an audio file using the Whisper tool",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "convert_markdown_to_html",
        "description": "Convert a Markdown file to HTML using Pandoc",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    },
    {
        "name": "filter_csv_to_json",
        "description": "Filter a CSV file based on a criteria and convert it to JSON",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []        
        }
    }
]

def invokeOpenAIProxy(sysMsg, usrMsg, useFunctionCall=False, data=None):
            # Set up the request body
        data = {
            "model":"gpt-4o-mini",
            "messages":[
                {"role": "system", "content": sysMsg},
                {"role": "user", "content": usrMsg}
            ]
        }

        if useFunctionCall:
            data["functions"]=function_descriptions
            data["function_call"]="auto"

        AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
        PROXY_URL = "https://aiproxy.sanand.workers.dev/openai"

        # Set up the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        # Make the POST request
        response = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=headers,
            json=data
        )

        return response

def execute_task(task):
    try:
        if not task:
            return {"error": "Invalid task description", "details": "Task description is required."}, 400

        sysMsg = "You are a task execution assistant. Parse the received task infromation and determine the appropriate action from the function description list."
        response = invokeOpenAIProxy(sysMsg, task, True)

        # Check the response
        if response.status_code == 200:
            result = response.json()
            print(result)

            message = result['choices'][0]['message']

            if 'function_call' in message:
                function_name = message['function_call']['name']
                arguments = message['function_call']['arguments']

                if function_name == 'run_datagen': #A1
                    return run_datagen()
                elif function_name == 'format_file': #A2
                    return format_file()
                elif function_name == 'count_number_of_days': #A3
                    return count_number_of_days(arguments[0])
                elif function_name == 'sort_contacts': #A4
                    return sort_contacts()
                elif function_name == 'extract_log_lines': #A5
                    return extract_log_lines()
                elif function_name == 'generate_markdown_index': #A6
                    return generate_markdown_index()
                elif function_name == 'extract_email_sender': #A7
                    return extract_email_sender()
                elif function_name == 'extract_credit_card_number': #A8
                    return extract_credit_card_number()
                elif function_name == 'find_similar_comments': #A9
                    return find_similar_comments()
                elif function_name == 'calculate_ticket_sales': #A10
                    return calculate_ticket_sales()
                elif function_name == 'fetch_data_from_api': #B3
                    return fetch_data_from_api(arguments[0])
                elif function_name == 'clone_git_repo': #B4
                    return clone_git_repo(arguments[0])
                elif function_name == 'run_sql_query': #B5
                    return run_sql_query(arguments[0], arguments[1], arguments[2])
                elif function_name == 'scrape_website': #B6
                    return scrape_website(arguments[0])
                elif function_name == 'process_image': #B7
                    return process_image()
                elif function_name == 'transcribe_audio': #B8
                    return transcribe_audio()
                elif function_name == 'convert_markdown_to_html': #B9
                    return convert_markdown_to_html()
                elif function_name == 'filter_csv_to_json': #B10
                    return filter_csv_to_json()
                else:
                    return jsonify({"error": "Function not implemented", "details": function_name}), 500
            else:
                content = message['content'].strip()
                return jsonify({"message": "Task executed successfully", "content": content}), 200
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

        return jsonify({"message": "Task execution initiated."}), 200  #Example success return

    except Exception as e:
        return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500

def run_datagen():
    try:
        subprocess.run(["uv", "install"], check=True)
        subprocess.run(["python3", "datagen.py", os.environ.get("email", "21f2000660@ds.study.iitm.ac.in")], check=True)
        return jsonify({"message": "Data generation successful."}), 200
    except Exception as e:
        return jsonify({"error": "Failed to run datagen", "details": str(e)}), 500

def format_file():
    try:
        subprocess.run(["npx", "prettier@3.4.2", "--write", "/data/format.md"], check=True)
        return jsonify({"message": "File formatted successfully."}), 200
    except Exception as e:
        return jsonify({"error": "Failed to format file", "details": str(e)}), 500

def count_number_of_days(day):
    dates_file = '/data/dates.txt'
    output_file = f"/data/dates-{day}.txt"

    try:
        with open(dates_file, 'r') as f:
            dates = f.readlines()

        # Count the number of days
        count = sum(1 for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)

        # Write the result to the output file
        with open(output_file, 'w') as f:
            f.write(str(count))

        print("Successfully counted Wednesdays and wrote to file.")
    except Exception as e:
        print(f"Error: {e}")

def sort_contacts():
    # Define the paths
    input_file = '/data/contacts.json'
    output_file = '/data/contacts-sorted.json'

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f'Input file {input_file} does not exist.')
    else:
        # Read the contacts from the file
        with open(input_file, 'r') as f:
            contacts = json.load(f)

        # Sort the contacts by last_name and then first_name
        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

        # Write the sorted contacts to the output file
        with open(output_file, 'w') as f:
            json.dump(sorted_contacts, f)

        print('Successfully sorted contacts and wrote to file.')

def extract_log_lines():
    # Define the path for log files and output file
    logs_path = '/data/logs/'
    output_file = '/data/logs-recent.txt'

    # Check if the logs directory exists
    if not os.path.exists(logs_path):
        print(f'Logs directory {logs_path} does not exist.')
    else:
        # Get the 10 most recent .log files
        log_files = sorted([f for f in os.listdir(logs_path) if f.endswith('.log')],
                        key=lambda x: os.path.getmtime(os.path.join(logs_path, x)),
                        reverse=True)[:10]

        # Write the first line of each log file to the output file
        with open(output_file, 'w') as f:
            for log_file in log_files:
                with open(os.path.join(logs_path, log_file), 'r') as lf:
                    first_line = lf.readline().strip()
                    f.write(first_line + '\n')

        print('Successfully extracted first lines of recent log files and wrote to file.')

def extract_email_sender():
    try:
        with open("/data/email.txt", "r") as f:
            email_content = f.read()

        sysMsg = "Extract the sender's email address from the given email content."
        usrMsg = email_content

        response = invokeOpenAIProxy(sysMsg, usrMsg)

        sender_email = response.json()["choices"][0]["message"]["content"].strip()
        with open("/data/email-sender.txt", "w") as f:
            f.write(sender_email)
        return {"message": "Email sender extracted successfully."}, 200
    except Exception as e:
        return {"error": "Failed to extract email sender", "details": str(e)}, 500

def extract_credit_card_number():
    try:
        sysMsg = "Extract the credit card number from the given image."
        userMsg = "Extract card number from /data/credit-card.png"

        response = invokeOpenAIProxy(sysMsg, userMsg)

        if response.status_code != 200:
            return {"error": "OpenAI API error", "details": response.text}, 500

        card_number = response.json()["choices"][0]["message"]["content"].strip()
        with open("/data/credit-card.txt", "w") as f:
            f.write(card_number.replace(" ", ""))
        return {"message": "Credit card number extracted successfully."}, 200
    except Exception as e:
        return {"error": "Failed to extract credit card number", "details": str(e)}, 500

def find_similar_comments():
    try:
        with open("/data/comments.txt", "r") as f:
            comments = f.readlines()
        
        sysMsg = "Find the most similar pair of comments."
        userMsg = "\n".join(comments)
        
        response = invokeOpenAIProxy(sysMsg, userMsg)

        if response.status_code != 200:
            return {"error": "OpenAI API error", "details": response.text}, 500

        similar_comments = response.json()["choices"][0]["message"]["content"].strip()
        with open("/data/comments-similar.txt", "w") as f:
            f.write(similar_comments)
        return {"message": "Most similar comments found successfully."}, 200
    except Exception as e:
        return {"error": "Failed to find similar comments", "details": str(e)}, 500

def generate_markdown_index():
    try:
        index = {}
        for file in os.listdir("/data/docs"):
            if file.endswith(".md"):
                with open(f"/data/docs/{file}", "r") as f:
                    for line in f:
                        if line.startswith("#"):
                            index[file] = line.strip("# ").strip()
                            break
        with open("/data/docs/index.json", "w") as f:
            json.dump(index, f)
        return jsonify({"message": "Markdown index generated successfully."}), 200
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/docs/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to generate markdown index", "details": str(e)}), 500

def calculate_ticket_sales():
    try:
        conn = sqlite3.connect("/data/ticket-sales.db")
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0] or 0  # Handle potential None value
        conn.close()
        with open("/data/ticket-sales-gold.txt", "w") as f:
            f.write(str(total_sales))
        return jsonify({"message": "Ticket sales calculated successfully."}), 200
    except sqlite3.Error as e:
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Database file not found", "details": "Ensure /data/ticket-sales.db exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to calculate ticket sales", "details": str(e)}), 500

def fetch_data_from_api(apiEndPoint):
    try:
        response = requests.get(apiEndPoint)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        with open("/data/api-data.json", "w") as f:
            json.dump(data, f)
        return jsonify({"message": "API data fetched and saved successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "API request failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500

def clone_git_repo(repoLink):
    try:
        subprocess.run(["git", "clone", repoLink, "/data/repo"], check=True)
        return jsonify({"message": "Git repository cloned successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Git clone failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to clone repository", "details": str(e)}), 500

def run_sql_query(dbLocation, fileLocation, queryToRun):
    try:
        conn = sqlite3.connect(dbLocation)
        cursor = conn.cursor()
        cursor.execute(queryToRun)
        result = cursor.fetchone()[0]
        conn.close()
        with open(fileLocation, "w") as f:
            f.write(str(result or 0))  # Handle potential None value
        return jsonify({"message": "SQL query executed successfully."}), 200
    except sqlite3.Error as e:
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Database file not found", "details": "Ensure /data/database.db exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to run SQL query", "details": str(e)}), 500

def scrape_website(websiteLink):
    try:
        response = requests.get(websiteLink)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open("/data/webpage.html", "w") as f:
            f.write(response.text)
        return jsonify({"message": "Website scraped successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Website scraping failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to scrape website", "details": str(e)}), 500

def process_image():
    try:
        img = Image.open("/data/image.png")
        width, height = img.size
        new_width = width // 2
        new_height = height // 2
        img = img.resize((new_width, new_height))
        img.save("/data/image-resized.png")
        return jsonify({"message": "Image processed successfully."}), 200
    except FileNotFoundError:
        return jsonify({"error": "Image file not found.", "details": "Ensure /data/image.png exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500

def transcribe_audio():
    try:
        subprocess.run(["whisper", "/data/audio.mp3", "--output", "/data/transcription.txt"], check=True)
        return jsonify({"message": "Audio transcribed successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Audio transcription failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found", "details": "Ensure /data/audio.mp3 exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to transcribe audio", "details": str(e)}), 500

def convert_markdown_to_html():
    try:
        subprocess.run(["pandoc", "/data/docs.md", "-o", "/data/docs.html"], check=True)
        return jsonify({"message": "Markdown converted to HTML successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Markdown conversion failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Markdown file not found", "details": "Ensure /data/docs.md exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to convert markdown", "details": str(e)}), 500

def filter_csv_to_json():
    try:
        with open("/data/data.csv", "r") as csv_file:
            lines = csv_file.readlines()
        filtered_data = [line for line in lines if "filter_criteria" in line]
        with open("/data/data.json", "w") as json_file:
            json
    except Exception as e:
        return jsonify({"error": "Failed to filter csv to json", "details": str(e)}), 500



@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task', '')
    return execute_task(task)  # Return the result of execute_task

@app.route('/read', methods=['GET'])
def read_file():
    path = request.args.get('path', '')
    if not path.startswith("/data/"):
        return jsonify({"error": "Access denied", "details": "Access outside /data/ is not allowed."}), 403

    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    with open(path, "r") as f:
        content = f.read()

    return content, 200

def main():
    app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == '__main__':
    main()
