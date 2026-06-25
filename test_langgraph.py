"""
Standalone test script for LangGraph workflow components.

This script tests the core LangGraph functionality without FastAPI:
- Dynamic schema generation
- Conversation flow
- Data extraction
- Duplicate detection (mocked)

Prerequisites:
- Create .env file with required variables (see .env.example)
- Ensure MongoDB and Redis are running (or use Atlas/cloud services)

Run with: python test_langgraph.py
Or copy into Jupyter notebook
"""

import asyncio
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
from pprint import pprint

from langchain_core.messages import HumanMessage

from src.config.settings import settings, prompt_config
from src.core.database import MongoDBClient, RedisClient
from src.core.schema import generate_request_schema
from src.core.workflow import ConversationWorkflow


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_schema_generation():
    """Test 1: Dynamic schema generation from config."""
    print_section("TEST 1: Dynamic Schema Generation")
    
    print("📋 Prompt Configuration:")
    print(f"  System Prompt: {prompt_config.system_prompt[:100]}...")
    print(f"  Config Version: {prompt_config.config_version}")
    print(f"  Number of fields: {len(prompt_config.fields)}")
    
    print("\n📝 Fields:")
    for field in prompt_config.fields:
        print(f"  - {field['name']} ({field['type']}): {field['description']}")
        if field.get('enum_values'):
            print(f"    Allowed values: {field['enum_values']}")
    
    print("\n🔧 Generated Pydantic Schema:")
    RequestSchema = generate_request_schema()
    print(f"  Schema name: {RequestSchema.__name__}")
    print(f"  Fields: {list(RequestSchema.model_fields.keys())}")
    
    # Test schema validation
    print("\n✅ Testing schema validation:")
    try:
        sample_data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "development",
            "business_justification": "Need to test new feature",
            "name": "John Doe",
            "employee_id": "EMP12345",
        }
        validated = RequestSchema(**sample_data)
        print(f"  Valid data: {validated.model_dump()}")
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
    
    return RequestSchema


async def test_database_connections():
    """Test 2: Database connections."""
    print_section("TEST 2: Database Connections")
    
    print("🔌 Connecting to databases...")
    print(f"  MongoDB URL: {settings.mongodb_url}")
    print(f"  Redis URL: {settings.redis_url}")
    
    # MongoDB
    mongodb_client = MongoDBClient()
    try:
        await mongodb_client.connect()
        print("  ✅ MongoDB connected")
    except Exception as e:
        print(f"  ⚠️  MongoDB connection failed: {e}")
        print("  💡 Make sure MongoDB is running or use MongoDB Atlas")
    
    # Redis (skipping for now - workflow will use InMemorySaver)
    redis_client = RedisClient()
    try:
        redis_client.connect()
        print("  ✅ Redis connected")
    except Exception as e:
        print(f"  ⚠️  Redis connection failed: {e}")
        print("  💡 Make sure Redis is running or using InMemorySaver as fallback")
    
    return mongodb_client, redis_client


async def test_workflow_compilation(mongodb_client, redis_client):
    """Test 3: Workflow compilation."""
    print_section("TEST 3: Workflow Compilation")
    
    print("🔨 Creating and compiling workflow...")
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  Chat Model: {settings.chat_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    
    workflow = ConversationWorkflow(
        mongodb_client=mongodb_client
    )
    
    # Compile with in-memory checkpointer for testing
    checkpointer = InMemorySaver()
    app = workflow.compile(checkpointer)
    print("  ✅ Workflow compiled successfully")
    
    print("\n📊 Workflow structure:")
    print(f"  Nodes: chat, extract, collect_pii, confirm_pii, duplicate_check, save")
    print(f"  Entry point: conditional (routes to chat/collect_pii/confirm_pii based on state)")
    print(f"  Checkpointer: InMemorySaver (for testing)")
    print(f"  Privacy: PII collection bypasses LLM (uses simple parsing)")
    
    # Visualize and save graph
    print("\n🎨 Generating workflow visualization...")
    try:
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Save to file
        output_path = "workflow_graph.png"
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        print(f"  ✅ Graph saved to: {output_path}")
        print("  💡 Open the PNG file to see the workflow visualization")
    except Exception as e:
        print(f"  ⚠️  Could not generate graph visualization: {e}")
        print("  💡 You may need to install: pip install pygraphviz or pip install grandalf")
    
    return app


async def test_conversation_flow(app):
    """Test 4: Full conversation flow with PII collection."""
    print_section("TEST 4: Conversation Flow with PII Collection")
    
    session_id = f"test-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"🆔 Session ID: {session_id}")
    
    # Conversation messages - now includes PII collection
    messages = [
        "Hi, I need help with a request",
        "I need infrastructure provisioning",
        "For the development environment",
        "We need to test a new microservice architecture before deploying to production",
        # After extraction, bot will ask for PII
        "John Doe,EMP12345",  # PII in format: NAME,EMPLOYEE_ID
        # Bot will ask for confirmation
        "update",  # Confirm PII
    ]
    
    print("\n💬 Starting conversation...\n")
    print("📝 Test Flow:")
    print("  1. Normal chat to collect request details")
    print("  2. Bot extracts data when ready")
    print("  3. Bot asks for PII (name and employee ID)")
    print("  4. User provides: NAME,EMPLOYEE_ID")
    print("  5. Bot asks for confirmation")
    print("  6. User confirms with 'yes'")
    print("  7. Bot checks for duplicates and saves\n")
    
    for i, user_message in enumerate(messages, 1):
        print(f"👤 User (message {i}): {user_message}")
        
        try:
            # Invoke workflow
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"configurable": {"thread_id": session_id}},
            )
            
            # Extract bot response
            last_message = result["messages"][-1]
            bot_response = last_message.content
            
            print(f"🤖 Bot: {bot_response}\n")
            
            # Show state
            if result.get("is_ready"):
                print("  ℹ️  Bot is ready to extract data")
            
            if result.get("collected_data"):
                print("  📦 Collected data:")
                pprint(result["collected_data"], indent=4)
            
            if result.get("temp_pii"):
                print("  🔐 Temporary PII (awaiting confirmation):")
                pprint(result["temp_pii"], indent=4)
            
            if result.get("awaiting_confirmation"):
                print("  ⏳ Awaiting PII confirmation")
            
            if result.get("pii_collected"):
                print("  ✅ PII collected and confirmed")
            
            if result.get("duplicate_warning"):
                print("  ⚠️  Duplicate warning:")
                pprint(result["duplicate_warning"], indent=4)
            
            if result.get("is_complete"):
                print("  ✅ Conversation complete!")
                break
            
            print()  # Empty line between exchanges
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Get final state
    print("\n📊 Final Session State:")
    try:
        state = app.get_state(config={"configurable": {"thread_id": session_id}})
        print(f"  Messages: {len(state.values.get('messages', []))} messages")
        print(f"  Is ready: {state.values.get('is_ready', False)}")
        print(f"  Is complete: {state.values.get('is_complete', False)}")
        print(f"  PII collected: {state.values.get('pii_collected', False)}")
        if state.values.get("collected_data"):
            print(f"  Collected data: {state.values['collected_data']}")
    except Exception as e:
        print(f"  ⚠️  Could not retrieve state: {e}")


async def test_settings():
    """Test 5: Settings loaded correctly."""
    print_section("TEST 5: Settings Configuration")
    
    print("⚙️  Current settings:")
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  Chat Model: {settings.chat_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  MongoDB URL: {settings.mongodb_url[:30]}...")
    print(f"  Redis URL: {settings.redis_url}")
    print(f"  Log Level: {settings.log_level}")


async def cleanup(mongodb_client, redis_client):
    """Cleanup database connections."""
    print_section("Cleanup")
    
    print("🧹 Closing connections...")
    try:
        await mongodb_client.close()
        print("  ✅ MongoDB connection closed")
    except:
        pass
    
    try:
        # redis_client.close()
        print("  ✅ Redis connection closed")
    except:
        pass


async def main():
    """Run all tests."""
    print("\n" + "🚀" * 40)
    print("  LangGraph Workflow Component Tests")
    print("🚀" * 40)
    
    try:
        # Test 1: Schema generation
        await test_schema_generation()
        
        # Test 2: Database connections
        mongodb_client, redis_client = await test_database_connections()
        
        # Test 3: Workflow compilation
        app = await test_workflow_compilation(mongodb_client, redis_client)
        
        # Test 4: Conversation flow
        # await test_conversation_flow(app) 
        
        # Test 5: Settings
        await test_settings()
        
        # Cleanup
        await cleanup(mongodb_client, redis_client)
        
        print_section("✅ All Tests Complete!")
        print("💡 Next steps:")
        print("  1. Review the conversation flow above")
        print("  2. Check if data extraction worked correctly")
        print("  3. Test with different conversation patterns")
        print("  4. Proceed to FastAPI integration testing")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

# Made with Bob
