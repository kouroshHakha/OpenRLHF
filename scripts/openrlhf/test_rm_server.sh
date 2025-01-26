
# Set variables
# PRED 1 should be classified as incorrect
export PRED1='To find the diameter of the image of a planet at the focus of the Magellan telescopes...[previous content]...\\[\n\\boxed{0.20}\n\\]'
export SOL1='6.5'

# PRED2 should be classified as correct
export PRED2='To determine the critical angle for the light to be internally reflected...[previous content]...\\boxed{41.8}\\) degrees.'
export SOL2='41.80'

# Execute curl command
curl -X POST http://localhost:5000/get_reward_math \
  -H "Content-Type: application/json" \
  -d '{
    "prediction": ["'"$PRED1"'", "'"$PRED2"'"],
    "solution": ["'"$SOL1"'", "'"$SOL2"'"]
  }'