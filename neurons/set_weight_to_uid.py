
import sys
import argparse
import bittensor as bt

def main():
    # 1) Parse the single argument for target_uid
    parser = argparse.ArgumentParser(
        description="Set weights on netuid=68 so that only target_uid has weight=1."
    )
    parser.add_argument('--target_uid', type=int, required=True,
                        help="The UID that will receive weight=1.0. Others = 0.0")
    parser.add_argument('--wallet_name', type=str, required=True,
                        help="The name of the wallet to use.")
    parser.add_argument('--wallet_hotkey', type=str, required=True,
                        help="The hotkey to use for the wallet.")

    args = parser.parse_args()

    NETUID = 68
    
    wallet = bt.wallet(
        name=args.wallet_name,  
        hotkey=args.wallet_hotkey, 
    )

    # Create Subtensor connection
    subtensor = bt.subtensor()

    # Download the metagraph for netuid=68
    metagraph = subtensor.metagraph(NETUID)

    # Check registration
    hotkey_ss58 = wallet.hotkey.ss58_address
    if hotkey_ss58 not in metagraph.hotkeys:
        print(f"Hotkey {hotkey_ss58} is not registered on netuid {NETUID}. Exiting.")
        sys.exit(1)

    # 2) Build the weight vector
    n = len(metagraph.uids)
    weights = [0.0] * n

    # Validate the user-provided target UID
    if not (0 <= args.target_uid < n):
        print(f"Error: target_uid {args.target_uid} out of range [0, {n-1}]. Exiting.")
        sys.exit(1)

    # Set the single weight
    weights[args.target_uid] = 1.0

    # 3) Send the weights to the chain
    print(f"Setting weight=1 on UID={args.target_uid} (netuid={NETUID}), 0 on others.")
    result = subtensor.set_weights(
        netuid=NETUID,
        wallet=wallet,
        uids=metagraph.uids,
        weights=weights,
        wait_for_inclusion=True
    )
    print(f"Result from set_weights: {result}")
    print("Done.")

if __name__ == "__main__":
    main()
