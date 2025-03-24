import asyncio
import os
import cognee
import logging
import glob
from cognee.api.v1.search import SearchType
from cognee.shared.utils import render_graph, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "INFORMATION-TECHNOLOGY")
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))[:5]  # Limit to first 5 files
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"- {os.path.basename(pdf_file)}")
    
    print("\nAdding PDFs to cognee:")
    # Add each PDF file to cognee
    for pdf_file in pdf_files:
        print(f"Processing {os.path.basename(pdf_file)}...")
        await cognee.add(pdf_file)
    
    # Use LLMs and cognee to create knowledge graph
    print("Starting cognify")
    await cognee.cognify()
    print("Cognify process complete.\n")

    await render_graph()

    # Example query - we can tailor this to the IT domain
    query_text = "Which candidate has successfully led data center consolidations, reducing operational costs, and can show experience managing multi-million-dollar IT budgets?"
    print(f"Searching cognee with query: '{query_text}'")

    search_results = await cognee.search(query_type=SearchType.COMPLETION, query_text=query_text)

    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
