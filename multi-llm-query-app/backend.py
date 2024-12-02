import sys, os, time, json, textwrap, polyllm, tiktoken
import multiprocessing as mp
from datetime import datetime
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"], "supports_credentials": True}})
CORS(app)

gpt4o_encoding = tiktoken.encoding_for_model("gpt-4o")
enc = gpt4o_encoding.encode("testing the encoding")
dec = gpt4o_encoding.decode(enc)
print("ENCTEST",enc,"DECTEST",dec)

@app.route('/')
def serve_benchmark():
    return send_file('bench05.html')

with open("models.txt") as f:
    model_info = {}
    for line in f:
        (model_name,cost_in,cost_out) = line.split()
        model_info[model_name] = (float(cost_in), float(cost_out))
    model_names = model_info.keys()

def mp_call_model(args):
    (prompt,model_name) = args
    prompt_enc = gpt4o_encoding.encode(prompt)  # for cost
    json_messages = [
        {
            "role": "user",
            "content": textwrap.dedent(f"""
                <prompt>
                {prompt}
                </prompt>
                Produce the result in JSON that matches this schema:
                    {{
                        "answer": "the answer to the prompt query",
                    }}
                """).strip()
        }
    ]

    in_cost = (len(prompt_enc) / 1_000_000) * model_info[model_name][0]
    stime = time.time()
    jresponse = polyllm.generate(model_name, json_messages, json_output=True)
    jresponse_enc = gpt4o_encoding.encode(jresponse)  # for cost
    out_cost = (len(jresponse_enc) / 1_000_000) * model_info[model_name][1]
    exec_time = time.time() - stime
    result = {
        "completion": jresponse,
        "model": model_name,
        "execTime": round(exec_time,3),
        "execCost": round(in_cost + out_cost,6),
    }
    print("DBGRESULT", result)
    return result

@app.route('/api/run-benchmark', methods=['POST'])
def run_benchmark():
    prompt = request.json.get('prompt', '')
    if not prompt:
        print("NO PROMPT PROVIDED")
        return jsonify({"error": "No prompt provided"}), 400
    print("PROMPT,", prompt)
    args = list( zip([prompt]*len(model_names), model_names) )
    pool = mp.Pool(processes=len(model_names))
    results = pool.map(mp_call_model, args)
    print("DBGRESULTS",results)

    json_results = jsonify(results)
    return json_results

@app.route('/api/save-benchmark', methods=['POST'])
def save_benchmark():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    # Create saved_benchmarks directory if it doesn't exist
    os.makedirs('saved_benchmarks', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'saved_benchmarks/benchmark_{timestamp}.json'
    
    # Save the data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
        
    return jsonify({"message": "Benchmark saved", "filename": filename})

@app.route('/api/get-model-names', methods=['GET'])
def get_model_names():
    return jsonify({"models": list(model_names)})

if __name__ == '__main__':
    app.run(debug=True, port=5555)  # 5000 sometimes used on macOS
