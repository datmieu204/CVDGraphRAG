"""
Three-layer architecture import script
Complete implementation of the paper's three-layer architecture: Bottom/Middle/Top
"""

import os
import sys
import argparse
from pathlib import Path

from logger_ import get_logger

logger = get_logger("three_layer_importer", log_file="logs/three_layer_importer.log")

from camel.storages import Neo4jGraph
from dataloader import load_high
from creat_graph_with_description import creat_metagraph_with_description
from import_umls_csv import import_umls_csv_to_neo4j
from utils import str_uuid, ref_link, add_sum

import gc

class ThreeLayerImporter:
    """Three-layer architecture importer"""
    
    def __init__(self, neo4j_url, neo4j_username, neo4j_password):
        """Initialization"""
        logger.info("\n" + "="*80)
        logger.info("Three-layer architecture knowledge graph importer")
        logger.info("="*80)

        # Connect to Neo4j
        logger.info("\n[Connecting to Neo4j]...")
        self.n4j = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        logger.info("✅ Neo4j connection successful")
        
        # Store GID for each layer
        self.layer_gids = {
            'bottom': [],
            'middle': [],
            'top': []
        }
    
    def clear_database(self):
        """Clear database (automatically clear, no confirmation needed)"""
        logger.info("\n[Clearing database]...")
        result = self.n4j.query("MATCH (n) RETURN count(n) as count")
        count = result[0]['count'] if result else 0
        logger.info(f"Current number of nodes: {count}")
        
        if count > 0:
            logger.info("Automatically clearing database...")
            self.n4j.query("MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared")
        else:
            logger.info("Database is already empty")

    def import_layer(self, layer_name: str, data_path: str, args):
        """
        Import one layer's data
        
        Args:
            layer_name: Layer name (bottom/middle/top)
            data_path: Data path
            args: Other parameters
        """
        logger.info("\n" + "="*80)
        logger.info(f"[{layer_name.upper()} layer] Starting import")
        logger.info(f"Data path: {data_path}")
        logger.info("="*80)

        data_path = Path(data_path)
        
        # Get all text files (including .txt and .csv)
        if data_path.is_file():
            files = [data_path]
        else:
            txt_files = list(data_path.glob("*.txt"))
            csv_files = list(data_path.glob("*.csv"))
            # Recursively find files in subdirectories
            txt_files.extend(data_path.rglob("*/*.txt"))
            csv_files.extend(data_path.rglob("*/*.csv"))
            files = txt_files + csv_files

        logger.info(f"\nFound {len(files)} files")

        # Classification statistics
        txt_count = sum(1 for f in files if f.suffix == '.txt')
        csv_count = sum(1 for f in files if f.suffix == '.csv')
        logger.info(f"  - TXT files: {txt_count} (free text)")
        logger.info(f"  - CSV files: {csv_count} (structured data)")

        # Process each file
        for idx, file_path in enumerate(files, 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"[File {idx}/{len(files)}] {file_path.name}")
            logger.info(f"{'─'*80}")

            # ========== New: checkpoint check ==========
            done_flag = file_path.with_suffix(file_path.suffix + ".done")
            if done_flag.exists():
                logger.info(f"⏭️  Skipping (completed): {file_path.name}")
                continue
            # =========================================

            try:
                gid = str_uuid()
                self.layer_gids[layer_name].append(gid)

                if file_path.suffix == '.csv':
                    logger.info(f"  [Type] Structured data (CSV)")
                    success = import_umls_csv_to_neo4j(str(file_path), gid, self.n4j)
                    if success:
                        summary_text = f"UMLS knowledge from {file_path.name}"
                        add_sum(self.n4j, summary_text, gid)
                    else:
                        logger.warning(f"⚠️  Processing failed: {file_path.name}")
                        self.layer_gids[layer_name].remove(gid)
                        continue

                else:
                    logger.info(f"  [Type] Free text (TXT)")
                    content = load_high(str(file_path))
                    if not content or len(content.strip()) < 50:
                        logger.warning(f"⚠️  Skip: Content too short")
                        self.layer_gids[layer_name].remove(gid)
                        continue

                    self.n4j = creat_metagraph_with_description(
                        args, content, gid, self.n4j
                    )

                # ========== New: Create .done file as checkpoint ==========
                done_flag.touch()
                logger.info(f"✅ Completed and recorded checkpoint: {done_flag.name}")
                # ========================================================

                gc.collect()

            except Exception as e:
                logger.error(f"❌ Error: {file_path.name} - {e}")
                import traceback
                traceback.print_exc()
                try:
                    self.layer_gids[layer_name].remove(gid)
                except:
                    pass
                continue


        logger.info(f"\n{'='*80}")
        logger.info(f"[{layer_name.upper()} layer] Import completed")
        logger.info(f"Imported {len(self.layer_gids[layer_name])} subgraphs")
        logger.info(f"{'='*80}")

    def create_trinity_links(self):
        """Create REFERENCE relationships between the three layers"""
        logger.info("\n" + "="*80)
        logger.info("[Trinity links] Starting to create cross-layer relationships")
        logger.info("="*80)
        
        total_links = 0
        
        # Bottom -> Middle
        logger.info("\n[Links] Bottom → Middle")
        for bottom_gid in self.layer_gids['bottom']:
            for middle_gid in self.layer_gids['middle']:
                try:
                    result = ref_link(self.n4j, bottom_gid, middle_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            logger.info(f"  ✅ {bottom_gid[:8]}... → {middle_gid[:8]}...: {count} links")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error: {e}")

        # Middle -> Top
        logger.info("\n[Links] Middle → Top")
        for middle_gid in self.layer_gids['middle']:
            for top_gid in self.layer_gids['top']:
                try:
                    result = ref_link(self.n4j, middle_gid, top_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            logger.info(f"  ✅ {middle_gid[:8]}... → {top_gid[:8]}...: {count} links")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error: {e}")

        logger.info(f"\n{'='*80}")
        logger.info(f"[Trinity links] Completed")
        logger.info(f"Created {total_links} REFERENCE relationships")
        logger.info(f"{'='*80}")

    def print_statistics(self):
        """Print statistics"""
        logger.info("\n" + "="*80)
        logger.info("[Statistics]")
        logger.info("="*80)
        
        # Node statistics
        result = self.n4j.query("MATCH (n) WHERE NOT n:Summary RETURN count(n) as count")
        node_count = result[0]['count'] if result else 0
        
        # Summary statistics
        result = self.n4j.query("MATCH (s:Summary) RETURN count(s) as count")
        summary_count = result[0]['count'] if result else 0
        
        # Relationship statistics
        result = self.n4j.query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result[0]['count'] if result else 0
        
        # REFERENCE statistics
        result = self.n4j.query("MATCH ()-[r:REFERENCE]->() RETURN count(r) as count")
        ref_count = result[0]['count'] if result else 0
        
        # Entity type statistics
        result = self.n4j.query("""
            MATCH (n)
            WHERE NOT n:Summary
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
            LIMIT 10
        """)

        logger.info(f"\nOverall statistics:")
        logger.info(f"  - Entity nodes: {node_count}")
        logger.info(f"  - Summary nodes: {summary_count}")
        logger.info(f"  - Total relationships: {rel_count}")
        logger.info(f"  - REFERENCE relationships: {ref_count}")

        logger.info(f"\nLayer statistics:")
        logger.info(f"  - Bottom layer: {len(self.layer_gids['bottom'])} subgraphs")
        logger.info(f"  - Middle layer: {len(self.layer_gids['middle'])} subgraphs")
        logger.info(f"  - Top layer: {len(self.layer_gids['top'])} subgraphs")

        logger.info(f"\nEntity types (top 10):")
        for item in result:
            logger.info(f"  - {item['type']}: {item['count']}")

        logger.info(f"\n{'='*80}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Three-layer architecture knowledge graph import')
    
    # Data paths
    parser.add_argument('--bottom', type=str, help='Bottom layer data path (medical dictionary)')
    parser.add_argument('--middle', type=str, help='Middle layer data path (diagnostic and treatment guidelines)')
    parser.add_argument('--top', type=str, help='Top layer data path (cases)')
    
    # Function switches
    parser.add_argument('--clear', action='store_true', help='Clear database')
    parser.add_argument('--trinity', action='store_true', help='Create Trinity relationships')
    parser.add_argument('--grained_chunk', action='store_true', help='Use fine-grained chunking')
    parser.add_argument('--ingraphmerge', action='store_true', help='Merge similar nodes within graph')
    
    # Neo4j configuration
    parser.add_argument('--neo4j-url', type=str, 
                       default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str, 
                       default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, 
                       default=os.getenv('NEO4J_PASSWORD'))
    
    args = parser.parse_args()
    
    # Check Neo4j password
    if not args.neo4j_password:
        logger.error("❌ Error: No Neo4j password provided")
        logger.info("Please set environment variable NEO4J_PASSWORD or use --neo4j-password parameter")
        sys.exit(1)
    
    # Initialize importer
    importer = ThreeLayerImporter(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password
    )
    
    # Clear database
    if args.clear:
        importer.clear_database()
    
    # Import each layer
    if args.bottom:
        importer.import_layer('bottom', args.bottom, args)
    
    if args.middle:
        importer.import_layer('middle', args.middle, args)
    
    if args.top:
        importer.import_layer('top', args.top, args)
    
    # Create Trinity relationships
    if args.trinity:
        importer.create_trinity_links()
    
    # Print statistics
    importer.print_statistics()

    logger.info("\n🎉 All tasks completed!")


if __name__ == '__main__':
    main()