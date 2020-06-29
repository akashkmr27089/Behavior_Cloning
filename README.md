# Behavior_Cloning
Behaviour Clonning On OpenAI Environment

### DAgger Algorithm For Behavior Cloning In Open AI Lunar Lander
1. Train Behavioral Policy Network from human data or any perfect System ( We used pre trained Nural Network Model of Lunar Lander as our Master Network) `D_collection`
2. Re-run the Behavioural Policy network on new simulation and colect the data.
3. Correctly label the generated data
4. Aggrigate the generated and corrected data with the `D_collection` and retrain
5. Repeat step *2-5* and measure the performance 


Special Thanks to : @nikhilbarhate99 for his PPO Model
