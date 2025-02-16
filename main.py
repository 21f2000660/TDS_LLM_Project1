"""
TDS Project - LLM based Assistant
"""

import os
import subprocess
import sqlite3
import json
import csv
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

function_descriptions = [
    {
        "name": "run_datagen",
        "description": "Install UV if not installed already, then fetch and run the datagen.py script from GitHub with the user's email as an argument.",
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
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path of input markdown file that needs to be formatted."
                }
            },
            "required": []        
        }
    },
    {
        "name": "count_number_of_days",
        "description": "Count the number of given days in a list of dates",
        "parameters": {
            "type": "object",
            "properties": {              
                "day": {
                    "type": "string",
                    "description": "The day that needs to be counted in the list of dates and format the day string in the following format 'Monday' or 'Wednesday'"
                },
                "input_file": {
                    "type": "string",
                    "description": "The path of the file that has the list of input dates that needs to be counted as per user request. This is not mandatory."
                },
                "output_file": {
                    "type": "string",
                    "description": "The counted days number should be written in the output file specified by the user with the file path. This is not mandatory."
                }
            },
            "required": ["day"]
        }
    },
    {
        "name": "sort_contacts",
        "description": "Sort contacts by last name and then by first name",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input json file that has unsorted contact information, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the output json file where the sorted contact information will be written, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }    
    },
    {
        "name": "extract_log_lines",
        "description": "Extract the first line from the 10 most recent log files",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path of the directory to identify all the logs files, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output text file on which the extracted log information will be written, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }
    },
    {
        "name": "generate_markdown_index",
        "description": "Generate an index of Markdown files and their first headings",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path of the directory to identify all the markdown files, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output json file on which the extracted index information with file name and title will be written as key-value pairs, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }
    },
    {
        "name": "extract_email_sender",
        "description": "Extract the sender's email address from an email content using GPT model",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input text file with the mail content to be read, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output text file on which the extracted mail sender id will be written, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }
    },
    {
        "name": "extract_credit_card_number",
        "description": "Extract credit card number from an image using GPT model",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input image file to be read, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output text file on which the extracted card number will be written, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }
    },
    {
        "name": "find_similar_comments",
        "description": "Find the most similar pair of comments using GPT model and embeddings",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input file to be read, sent by the user under the task arg. This is not mandatory"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output text file on which the resulting similar comments will be written, sent by the user under the task arg. This is not mandatory"
                }
            },
            "required": []        
        }
    },
    {
        "name": "calculate_ticket_sales", 
        "description": "Calculate total sales for Gold tickets from a SQLite database",
        "parameters": {
            "type": "object",
            "properties": {
                "sqlQuery": {
                    "type": "string",
                    "description": "This is the SQL query to be executed on the database, sent by the user under the task arg"
                },
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input database to be read, sent by the user under the task arg"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output file on which the result will be written, sent by the user under the task arg"
                }
            },
            "required": ["sqlQuery", "input_file", "output_file"]     
        }
    },
    {
        "name": "fetch_data_from_api",
        "description": "Fetch data from an API and save it as JSON",
        "parameters": {
            "type": "object",
            "properties": {
                "apiEndPoint": {
                    "type": "string",
                    "description": "The data will be fetched from the provided API endpoint"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the output text file in which the fetched response will be dumped"
                }
            },
            "required": ["apiEndPoint", "output_file"]
        }
    },
    {
        "name": "clone_git_repo",
        "description": "Clone a Git repository to a specific directory",
        "parameters": {
            "type": "object",
            "properties": {
                "repoLink": {
                    "type": "string",
                    "description": "The original task description (unused in this function)"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location output directory where the repo will be cloned"
                }
            },
            "required": ["repoLink", "output_file"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Execute a SQL query on the database file provided",
        "parameters": {
            "type": "object",
            "properties": {
                "sqlQuery": {
                    "type": "string",
                    "description": "This is the SQL query to be executed on the database, sent by the user under the task arg"
                },
                "input_file": {
                    "type": "string",
                    "description": "The location and name of the input database to be read, sent by the user under the task arg"
                },
                "output_file": {
                    "type": "string",
                    "description": "The location and name of the new output file on which the result will be written, sent by the user under the task arg"
                }
            },
            "required": ["sqlQuery", "input_file", "output_file"]
        }
    },
    {
        "name": "scrape_website",
        "description": "Scrape a website and save its content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "websiteLink": {
                    "type": "string",
                    "description": "The website url to be scrapped"
                },
                "output_file": {
                    "type": "string",
                    "description": "The path of the output text file. The scrapped text shall retain English language or the original language of the website"
                }
            },
            "required": ["websiteLink", "output_file"]
        }
    },
    {
        "name": "process_image",
        "description": "Resize an image to the user requirements",
        "parameters": {
            "type": "object",
            "properties": {
                "new_width": {
                    "type": "string",
                    "description": "The new width must be calculated in pixels if a percetange is given and no direct number is given or prefer the direct number"
                },
                "new_height": {
                    "type": "string",
                    "description": "The new height must be calculated in pixels if a percetange is given and no direct number is given or prefer the direct number"
                },
                "input_file": {
                    "type": "string",
                    "description": "The path of the input image file. Get the actual dimensions of the input image if possible"
                }
            },
            "required": ["new_width", "new_height", "input_file"]
        }
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe an audio file using the Whisper tool",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path of the input audio file. Must be in format supported by Whisper tool preferably MP3"
                },
                "output_file": {
                    "type": "string",
                    "description": "The path of the output text file. The transcription text shall retain English language or the original language of the audio if fully supported"
                }
            },
            "required": ["input_file", "output_file"]
        }
    },
    {
        "name": "convert_markdown_to_html",
        "description": "Convert a Markdown file to HTML using Pandoc",
        "parameters": {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "The path of the markdown file that will be converted"
                },
                "output_file": {
                    "type": "string",
                    "description": "The path of the output html file after coversion"
                }
            },
            "required": ["input_file", "output_file"]     
        }
    },
    {
        "name": "filter_csv_to_json",
        "description": "Filter a CSV file based on a criteria and convert it to JSON",
        "parameters": {
            "type": "object",
            "properties": {              
                "filter_criteria": {
                    "type": "string",
                    "description": "The crieteria used to filter the csv entries"
                },
                "input_file": {
                    "type": "string",
                    "description": "Accept only csv input files that will be filtered"
                }
            },
            "required": ["filter_criteria", "input_file"]   
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
            json_string = json.dumps(result, indent=2)
            message = result['choices'][0]['message']
            
            if 'function_call' in message:
                function_name = message['function_call']['name']
                arguments = message['function_call']['arguments']
                parsed_arguments = json.loads(arguments)

                # Check if the input file 
                input_file = parsed_arguments.get("input_file")
                if input_file is not None:
                    if not os.path.exists(input_file):
                        return jsonify({"error": f"Input file {input_file} does not exist."}), 404
                    if not input_file.startswith("/data/"):
                        return jsonify({"error": "Access denied", "details": "Access outside /data/ is not allowed."}), 403
            
                if function_name is not None:
                    try:
                        choosen_function = eval(function_name)
                        print(parsed_arguments)
                        res = choosen_function(**parsed_arguments)
                        return res  #Example success return
                    except Exception as e:
                        return jsonify({"error": "Function call invoke failed", "details": str(e)}), 500
                else:
                    return jsonify({"error": "Function call not implemented", "details": function_name}), 500
            else:
                content = message['content'].strip()
                return jsonify({"error": "Function call failed", "details": str(content)}), 500
        else:
            return jsonify({"error": "Open AI API invoke failed", "details": str(response.text)}), response.status_code
    except Exception as e:
        return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500

def run_datagen():
    try:
        # Check if uv is installed
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Install uv if not present
            subprocess.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"], check=True, shell=True)
        
        # Fetch the Python script from GitHub
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        response = requests.get(url)
        response.raise_for_status()
        script_content = response.text
        
        # Get the email from environment variable or use default
        email = os.environ.get("email", "21f2000660@ds.study.iitm.ac.in")
        
        # Execute the script using uv run
        result = subprocess.run(
            ["uv", "run", "python3", "-c", script_content, email],
            check=True,
            capture_output=True,
            text=True
        )
        
        return {"message": "Data generation successful.", "output": result.stdout}, 200
    except requests.RequestException as e:
        return {"error": "Failed to fetch the script", "details": str(e)}, 500
    except subprocess.CalledProcessError as e:
        return {"error": "Failed to run datagen", "details": e.stderr}, 500
    except Exception as e:
        return {"error": "Unexpected error", "details": str(e)}, 500

def format_file(input_file = None):
    if input_file is None:
        input_file = "/data/format.md"
    try:
        subprocess.run(["npx", "prettier@3.4.2", "--write", input_file], check=True)
        return jsonify({"message": "File formatted successfully."}), 200
    except Exception as e:
        return jsonify({"error": "Failed to format file", "details": str(e)}), 500

def count_number_of_days(day, input_file = None, output_file = None):
    if input_file is None:
        input_file = "/data/dates.txt"
    if output_file is None:
        output_file = f"/data/dates-{day.lower()}s.txt"
    try:
        with open(input_file, 'r') as f:
            dates = f.read()
            dates= dates.replace('\n', '')

        
        sysMsg = f"Parse the dates into day, find the occurence of {day} and return only the count in numeral and no other words."
        response = invokeOpenAIProxy(sysMsg, dates)
        result = response.json()
        count = result['choices'][0]['message']['content'].strip()

        # Write the result to the output file
        with open(output_file, 'w') as f:
            f.write(str(count))

        return jsonify({"message": f"Successfully counted {day} and wrote to file."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to count the occurences of {day} in the file", "details": str(e)}), 500

def sort_contacts(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = '/data/contacts.json'
    if output_file is None:
        output_file = '/data/contacts-sorted.json'
    
    try:
        # Read the contacts from the file
        with open(input_file, 'r') as f:
            contacts = json.load(f)

        # Sort the contacts by last_name and then first_name
        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

        # Write the sorted contacts to the output file
        with open(output_file, 'w') as f:
            json.dump(sorted_contacts, f)

        return jsonify({"message": "Successfully sorted contacts and wrote to file."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to sort contacts in the file", "details": str(e)}), 500

def extract_log_lines(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = '/data/logs/'
    if output_file is None:
        output_file = '/data/logs-recent.txt'

    try:
        # Get all .log files in the input directory
        log_files = [f for f in os.listdir(input_file) if f.endswith('.log')]
        
        # Sort log files by modification time, most recent first
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_file, x)), reverse=True)
        
        # Take only the 10 most recent files
        recent_logs = log_files[:10]
        
        # Write the first line of each recent log file to the output file
        with open(output_file, 'w') as out_file:
            for log_file in recent_logs:
                with open(os.path.join(input_file, log_file), 'r') as in_file:
                    first_line = in_file.readline().strip()
                    out_file.write(first_line + '\n')

        return jsonify({"message": "Successfully extracted first lines of recent log files and wrote to file."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to sort contacts in the file", "details": str(e)}), 500

def extract_email_sender(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = "/data/email.txt"
    if output_file is None:
        output_file = "/data/email-sender.txt"

    try:
        with open(input_file, "r") as f:
            email_content = f.read()

        sysMsg = "Extract the from or sender's email address from the given email content and send only the email address in the response, no other words"
        usrMsg = email_content

        response = invokeOpenAIProxy(sysMsg, usrMsg)

        sender_email = response.json()["choices"][0]["message"]["content"].strip()
        with open(output_file, "w") as f:
            f.write(sender_email)
        return {"message": "Email sender extracted successfully."}, 200
    except Exception as e:
        return {"error": "Failed to extract email sender", "details": str(e)}, 500

def extract_credit_card_number(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = "/data/credit-card.png"
    if output_file is None:
        output_file = "/data/credit-card.txt"
    try:
        sysMsg = "Extract the credit card number from the given image."
        userMsg = f"Extract card number from {input_file}"

        response = invokeOpenAIProxy(sysMsg, userMsg)

        if response.status_code != 200:
            return {"error": "OpenAI API error", "details": response.text}, 500

        card_number = response.json()["choices"][0]["message"]["content"].strip()
        with open(output_file, "w") as f:
            f.write(card_number.replace(" ", ""))
        return {"message": "Credit card number extracted successfully."}, 200
    except Exception as e:
        return {"error": "Failed to extract credit card number", "details": str(e)}, 500

def find_similar_comments(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = "/data/comments.txt"
    if output_file is None:
        output_file = "/data/comments-similar.txt"
    try:
        with open(input_file, "r") as f:
            comments = [line.strip() for line in f if line.strip()]
        
        # Initialize OpenAI Proxy client
        AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
        PROXY_URL = "https://aiproxy.sanand.workers.dev/openai"

        # Set up the headers and data
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }

        data = {
            "model": "gpt-4o-mini",
            "encoding_format": "float"
        }

        # Get embeddings for all comments
        embeddings = []
        for comment in comments:
            data["input"] = comment
            response = requests.post(
                f"{PROXY_URL}/v1/embeddings",
                headers=headers,
                json=data
            )
            embeddings.append(response.data[0].embedding)
        
        # Convert to numpy array for efficient computation
        embeddings_array = np.array(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Find the most similar pair (excluding self-similarity)
        np.fill_diagonal(similarity_matrix, -1)  # Exclude diagonal
        max_similarity_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        
        # Get the most similar comments
        similar_comment1 = comments[max_similarity_index[0]]
        similar_comment2 = comments[max_similarity_index[1]]
        
        # Write to output file
        with open(output_file, "w") as f:
            f.write(f"{similar_comment1}\n{similar_comment2}")
        return {"message": "Most similar comments found successfully."}, 200
    except Exception as e:
        return {"error": "Failed to find similar comments", "details": str(e)}, 500

def generate_markdown_index(input_file = None, output_file = None):
    # Define the paths
    if input_file is None:
        input_file = "/data/docs"
    if output_file is None:
        output_file = "/data/docs/index.json"
    try:
        index = {}
        index = {}
        for root, _, files in os.walk(input_file):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, input_file)
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("# "):
                                index[relative_path] = line.strip("# ").strip()
                                break
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        return jsonify({"message": "Markdown index generated successfully."}), 200
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": f"Ensure {input_file}/{file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to generate markdown index", "details": str(e)}), 500

def calculate_ticket_sales(sqlQuery = None, input_file = None, output_file = None):
    # Define the paths
    if sqlQuery is None:
        sqlQuery = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
    if input_file is None:
        input_file = "/data/ticket-sales.db"
    if output_file is None:
        output_file = "/data/ticket-sales-gold.txt"

    try:
        conn = sqlite3.connect(input_file)
        cursor = conn.cursor()
        cursor.execute(sqlQuery)
        total_sales = cursor.fetchone()[0] or 0  # Handle potential None value
        conn.close()
        with open(output_file, "w") as f:
            f.write(str(total_sales))
        return jsonify({"message": "Ticket sales calculated successfully."}), 200
    except sqlite3.Error as e:
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Database file not found", "details": f"Ensure {input_file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to calculate ticket sales", "details": str(e)}), 500

def fetch_data_from_api(apiEndPoint, output_file = None):
    # Define the paths
    if output_file is None:
        output_file = "/data/api-result-json.json"
    try:
        response = requests.get(apiEndPoint)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        with open(output_file, "w") as f:
            json.dump(data, f)
        return jsonify({"message": "API data fetched and saved successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "API request failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500

def clone_git_repo(repoLink, output_file = None):
    if output_file is None:
        output_file = "/data"
    try:
        subprocess.run(["git", "clone", repoLink, output_file], check=True)
        return jsonify({"message": "Git repository cloned successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Git clone failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to clone repository", "details": str(e)}), 500

def run_sql_query(sqlQuery, input_file, output_file):
    # Define the paths
    try:
        conn = sqlite3.connect(input_file)
        cursor = conn.cursor()
        cursor.execute(sqlQuery)
        result = cursor.fetchone()[0]
        conn.close()
        with open(output_file, "w") as f:
            f.write(str(result or 0))  # Handle potential None value
        return jsonify({"message": "SQL query executed successfully."}), 200
    except sqlite3.Error as e:
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Database file not found", "details": f"Ensure {input_file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to run SQL query", "details": str(e)}), 500

def scrape_website(websiteLink, output_file):
    try:
        response = requests.get(websiteLink)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(output_file, "w") as f:
            f.write(response.text)
        return jsonify({"message": "Website scraped successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Website scraping failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Directory not found", "details": "Ensure /data/ exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to scrape website", "details": str(e)}), 500

def process_image(new_width, new_height, input_file):
    try:
        img = Image.open(input_file)
        img = img.resize((new_width, new_height))
        img.save(input_file)
        return jsonify({"message": "Image processed successfully."}), 200
    except FileNotFoundError:
        return jsonify({"error": "Image file not found.", "details": f"Ensure {input_file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500

def transcribe_audio(input_file, output_file):
    try:
        subprocess.run(["whisper", input_file, "--output", output_file], check=True)
        return jsonify({"message": "Audio transcribed successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Audio transcription failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found", "details": f"Ensure {input_file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to transcribe audio", "details": str(e)}), 500

def convert_markdown_to_html(input_file, output_file):
    try:
        subprocess.run(["pandoc", input_file, "-o", output_file], check=True)
        return jsonify({"message": "Markdown converted to HTML successfully."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Markdown conversion failed", "details": str(e)}), 500
    except FileNotFoundError:
        return jsonify({"error": "Markdown file not found", "details": f"Ensure {input_file} exists."}), 404
    except Exception as e:
        return jsonify({"error": "Failed to convert markdown", "details": str(e)}), 500

def filter_csv_to_json(filter_criteria, input_file):
    try:
        # Read and filter CSV data
        filtered_data = []
        with open(input_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if any(filter_criteria.lower() in value.lower() for value in row.values()):
                    filtered_data.append(row)

        # Return filtered data as JSON
        return jsonify(filtered_data)
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
