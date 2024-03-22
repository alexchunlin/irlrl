#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiManager.h>
#include <Servo.h>

// This is the chassis(main) robot code, running on an esp32 MotorGo board
// It has the following goals:
//   - recieves UDP encoder data from the base_rig node (running on esp32)
//   - revieves TCP servo commands from the python wifi comms bridge (running on computer)
//   - sends TCP encoder data to the python wifi comms bridge
//   - interprets servo commands and writes according gait to servos

const int servoPins[] = {47, 38, 39, 40};
const int numServos = sizeof(servoPins) / sizeof(int);
Servo servos[numServos];
// map servo 1, 2, 3, 4 to left_hip, left_knee, right_hip, right_knee for readability
const int left_hip = 0;
const int left_knee = 1;
const int right_hip = 2;
const int right_knee = 3;


WiFiServer server(8080); // Set the desired port number

// struct for servo command data combined into one uint32_t for easy transfer
union ServoCommand {
    struct {
        uint8_t left_hip_deg;
        uint8_t left_knee_deg;
        uint8_t right_hip_deg;
        uint8_t right_knee_deg;
    };
    uint8_t servo_pos_commands[numServos];
};
// struct for data to transfer to the python wifi comms bridge
struct ResponseData {
    union{
        struct {
            float boom_vel_rad_s;
            float boom_pos_rad;
        };
        uint32_t raw;
    };
};

WiFiManager wm;
ServoCommand last_servo_command;
ResponseData response_to_rl_agent;

// servo writer function
void writeServoCommand(ServoCommand command) {
    servos[left_hip].write(command.left_hip_deg);
    servos[left_knee].write(command.left_knee_deg);
    servos[right_hip].write(command.right_hip_deg);
    servos[right_knee].write(command.right_knee_deg);
}

void setup() {
    Serial.begin(115200);

    // Automatically connect to WiFi using WiFiManager
    bool res = wm.autoConnect("AutoConnectAP", "password"); // Replace with your desired AP name and password
    if (!res) {
        Serial.println("Failed to connect to WiFi");
        ESP.restart();
    } else {
        Serial.println("Connected to WiFi");
    }

    server.begin();
    Serial.println("Server started");

    // Initialize servos
    for (int i = 0; i < numServos; i++) {
        servos[i].attach(servoPins[i]);
        int pos = 10; // near ~ 0 ish start position
        servos[i].write(pos);
    }
}
// 
void loop() {
    WiFiClient client = server.available();
    if (client) {
        if (client.available() >= sizeof(ServoCommand)) {
            // Read the servo command
            client.read((uint8_t*)&last_servo_command, sizeof(ServoCommand));

            // Write the servo command to the servos using our function
            writeServoCommand(last_servo_command);

            
            client.write((uint8_t*)&response_to_rl_agent, sizeof(float));
        }
        client.stop();
    } else {
        delay(1000);
        Serial.println("No client connected");
         for (int i = 0; i < numServos && i < MAX_SERVOS; i++) {
        int pos = random(10, 170);
        servos[i].write(pos);
    }
    }
}