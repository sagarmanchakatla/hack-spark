from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime
import uuid
import os

app = Flask(__name__)
CORS(app)

# Connect to MongoDB Atlas
# Replace with your MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://sagarmanchakatla02:XM3zr3uZjyEGkaY7@cluster0.zmxvk.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client['health_data_db']
collection = db['health_metrics']

@app.route('/give_info', methods=['POST'])
def give_info():
    data = request.get_json(force=True)
    print(data)
    # Process the incoming data
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Check the data format
    if isinstance(data, list):
        # Handle detailed health metrics format (list of individual measurements)
        stored_records = []
        for record in data:
            # Generate an ID if not provided
            if 'uuid' not in record:
                record['uuid'] = str(uuid.uuid4())
            
            # Convert date strings to datetime objects
            if 'dateFrom' in record:
                record['dateFrom'] = datetime.datetime.fromisoformat(record['dateFrom'].replace('Z', '+00:00'))
            if 'dateTo' in record:
                record['dateTo'] = datetime.datetime.fromisoformat(record['dateTo'].replace('Z', '+00:00'))
            
            # Store in MongoDB
            record_id = collection.insert_one(record).inserted_id
            stored_records.append(str(record_id))
        
        return jsonify({
            "success": True,
            "message": f"Stored {len(stored_records)} health metrics",
            "record_ids": stored_records
        })
    
    elif isinstance(data, dict):
        # Handle summary health data format
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.datetime.utcnow()
        
        # Add UUID if not present
        if 'uuid' not in data:
            data['uuid'] = str(uuid.uuid4())
        
        # Store in MongoDB
        record_id = collection.insert_one(data).inserted_id
        
        return jsonify({
            "success": True,
            "message": "Health summary data stored successfully",
            "record_id": str(record_id)
        })
    
    else:
        return jsonify({"error": "Invalid data format"}), 400

# Route to retrieve health data
@app.route('/get_health_data', methods=['GET'])
def get_health_data():
    # Get query parameters
    data_type = request.args.get('type')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Build query
    query = {}
    if data_type:
        query['type'] = data_type
    
    if start_date and end_date:
        start = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        query['dateFrom'] = {'$gte': start}
        query['dateTo'] = {'$lte': end}
    
    # Execute query
    results = list(collection.find(query, {'_id': 0}))
    
    # Convert datetime objects to ISO format strings for JSON serialization
    for result in results:
        if 'dateFrom' in result and isinstance(result['dateFrom'], datetime.datetime):
            result['dateFrom'] = result['dateFrom'].isoformat()
        if 'dateTo' in result and isinstance(result['dateTo'], datetime.datetime):
            result['dateTo'] = result['dateTo'].isoformat()
    
    return jsonify(results)

# Route to get daily summary
@app.route('/daily_summary', methods=['GET'])
def daily_summary():
    date = request.args.get('date', datetime.datetime.utcnow().strftime('%Y-%m-%d'))
    
    # Parse the date string
    try:
        target_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_day = target_date + datetime.timedelta(days=1)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    # Query for health data on the specified date
    pipeline = [
        {
            '$match': {
                'dateFrom': {
                    '$gte': target_date,
                    '$lt': next_day
                }
            }
        },
        {
            '$group': {
                '_id': '$type',
                'average': {'$avg': '$value.numericValue'},
                'max': {'$max': '$value.numericValue'},
                'min': {'$min': '$value.numericValue'},
                'count': {'$sum': 1},
                'unit': {'$first': '$unit'}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    return jsonify({
        'date': date,
        'metrics': results
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)