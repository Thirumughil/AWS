
import json
import os
import boto3
from decimal import Decimal

# Connect to the existing DynamoDB table.
dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.environ.get("TABLE_NAME", "HeatRiskPredictions")
table = dynamodb.Table(TABLE_NAME)


def calculate_risk(temperature, humidity):
    """Calculate the project's heat-risk category."""
    if temperature >= 40 and humidity >= 60:
        return "SEVERE"
    elif temperature >= 40 or (temperature >= 35 and humidity >= 60):
        return "HIGH"
    elif temperature >= 35 or (temperature >= 30 and humidity >= 70):
        return "MODERATE"
    else:
        return "LOW"


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }


def lambda_handler(event, context):
    try:
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            return response(400, {
                "error": "Request body must be a JSON object"
            })

        location_id = body.get("location_id")
        if not isinstance(location_id, str) or not location_id.strip():
            return response(400, {
                "error": "location_id is required"
            })

        try:
            temperature = float(body["temperature"])
            humidity = float(body["humidity"])
        except (KeyError, TypeError, ValueError):
            return response(400, {
                "error": "temperature and humidity must be numbers"
            })

        if not (-50 <= temperature <= 70):
            return response(400, {
                "error": "temperature must be between -50 and 70 Celsius"
            })

        if not (0 <= humidity <= 100):
            return response(400, {
                "error": "humidity must be between 0 and 100 percent"
            })

        risk_level = calculate_risk(temperature, humidity)

        table.put_item(
            Item={
                "location_id": location_id.strip(),
                "temperature": Decimal(str(temperature)),
                "humidity": Decimal(str(humidity)),
                "risk_level": risk_level
            }
        )

        return response(200, {
            "location_id": location_id.strip(),
            "temperature": temperature,
            "humidity": humidity,
            "risk_level": risk_level,
            "message": "Saved to DynamoDB"
        })

    except json.JSONDecodeError:
        return response(400, {
            "error": "Request body contains invalid JSON"
        })

    except Exception:
        print("Unexpected error while processing heat-risk request")
        import traceback
        traceback.print_exc()

        return response(500, {
            "error": "Failed to calculate or save heat risk"
        })