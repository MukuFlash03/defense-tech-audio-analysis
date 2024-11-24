from restack_ai.function import function, log
from dataclasses import dataclass
import os
import base64
from dotenv import load_dotenv
import psycopg2
from psycopg2 import Error
from datetime import datetime
import json

load_dotenv()

@dataclass
class FunctionInputParams:
    conversation_analysis: str

postgres_host = os.getenv("POSTGRES_HOST")
postgres_database = os.getenv("POSTGRES_DATABASE")
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")

def connect_to_postgres():
    try:
        connection = psycopg2.connect(
            host=postgres_host,
            database=postgres_database,
            user=postgres_user,
            password=postgres_password
        )
        log.info("Successfully connected to PostgreSQL database")
        return connection
    except (Exception, Error) as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None

@function.defn()
async def write_to_audio_table(conversation_analysis: FunctionInputParams):
    try:
        log.info("write_to_audio_table function started", input=conversation_analysis)
        
        log.info("Before connecting to PostgreSQL database")
        connection = connect_to_postgres()
        cursor = connection.cursor()
        log.info("After connecting to PostgreSQL database")
        
        data = json.loads(conversation_analysis.conversation_analysis)

        # insert_query = """
        #     INSERT INTO conversation_analysis (
        #         priority_level, risk_assessment, key_insights, 
        #         critical_entities, locations_mentioned, sentiment_summary,
        #         source_reliability, information_credibility, recommended_actions,
        #         entity_relationships, speakers, conversation_duration, analyzed_at
        #     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        # """
        
        # cursor.execute(insert_query, (
        #     data['priority_level'],
        #     data['risk_assessment'],
        #     data['key_insights'],
        #     data['critical_entities'],
        #     data['locations_mentioned'], 
        #     data['sentiment_summary'],
        #     data['source_reliability'],
        #     data['information_credibility'],
        #     data['recommended_actions'],
        #     data['entity_relationships'],
        #     data['speakers'],
        #     data['conversation_duration'],
        #     data['analyzed_at']
        # ))

        insert_query = """
            INSERT INTO conversation_analysis (
                priority_level, risk_assessment, key_insights, 
                critical_entities, locations_mentioned, sentiment_summary,
                source_reliability, information_credibility, recommended_actions,
                entity_relationships, speakers, conversation_duration, analyzed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convert arrays to JSON strings for JSON type columns
        cursor.execute(insert_query, (
            data['priority_level'],
            data['risk_assessment'],
            data['key_insights'],
            json.dumps(data['critical_entities']),  # JSON column
            json.dumps(data['locations_mentioned']),  # JSON column
            data['sentiment_summary'],
            data['source_reliability'],
            data['information_credibility'],
            json.dumps(data['recommended_actions']),  # JSON column
            data['entity_relationships'],
            json.dumps(data['speakers']),  # JSON column
            data['conversation_duration'],
            datetime.fromisoformat(data['analyzed_at'].replace('Z', '+00:00'))
        ))

        connection.commit()
        # print(f"Successfully inserted {len(data)} rows into {table_name}")

        log.info("write_to_audio_table function completed")        
        return "Successfully wrote analysis to database"
        
    except (Exception, Error) as error:
        log.info("Error writing to table", error=str(error))
        connection.rollback()  # Roll back in case of error
        log.error("write_to_audio_table function failed", error=error)
        raise error
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            log.info("PostgreSQL connection is closed")
            
@function.defn()
async def read_from_audio_table():
    try:
        log.info("read_from_table function started")

        log.info("Before connecting to PostgreSQL database")
        connection = connect_to_postgres()
        cursor = connection.cursor()
        log.info("After connecting to PostgreSQL database")
        
        query = f"SELECT * FROM conversation_analysis"

        cursor.execute(query)
        
        results = cursor.fetchall()
        
        column_names = [desc[0] for desc in cursor.description]
        print(f"Retrieved {len(results)} rows from conversation_analysis table")
        print(f"Columns: {column_names}")
        print("Results from DB:\n", results)

        formatted_results = []
        for row in results:
            formatted_row = list(row)
            for i, value in enumerate(formatted_row):
                if isinstance(value, datetime):
                    formatted_row[i] = value.isoformat()
            formatted_results.append(tuple(formatted_row))
            
        return formatted_results
        
    except (Exception, Error) as error:
        print(f"Error reading from table: {error}")
        return []

    finally:
        if cursor:
            cursor.close()
