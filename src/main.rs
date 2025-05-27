use anyhow::Result;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use aws_config::BehaviorVersion;
use aws_sdk_rekognition::{
    types::{Image, QualityFilter},
    Client as RekognitionClient,
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    sync::{Arc, Mutex},
};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessLog {
    timestamp: DateTime<Utc>,
    action: String,
    person_name: Option<String>,
    confidence: Option<f32>,
    access_granted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuthorizedPerson {
    name: String,
    face_id: String,
    external_image_id: String,
    added_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct AppState {
    rekognition_client: RekognitionClient,
    collection_id: String,
    access_log: Arc<Mutex<Vec<AccessLog>>>,
    authorized_people: Arc<Mutex<HashMap<String, AuthorizedPerson>>>,
    esp32_cam_url: String,
    pico2_door_url: String,
    confidence_threshold: f32,
}

#[derive(Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct AccessCheckResponse {
    access_granted: bool,
    person_name: Option<String>,
    confidence: Option<f32>,
    timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
struct AddPersonResponse {
    face_id: String,
    message: String,
}

impl AppState {
    async fn new() -> Result<Self> {
        // Load environment variables FIRST
        dotenvy::dotenv().expect("Failed to load .env file");
        
        // Verify credentials are loaded
        let aws_key = env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let aws_secret = env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");
        let aws_region = env::var("AWS_REGION").expect("AWS_REGION must be set");
        
        info!("🔑 AWS Key: {}...", &aws_key[..8]);
        info!("🌍 AWS Region: {}", aws_region);
        
        info!("🦀 Initializing Rust AWS Rekognition Door Lock...");
        
        let config = aws_config::defaults(BehaviorVersion::latest())
            .load()
            .await;
        
        let rekognition_client = RekognitionClient::new(&config);
        let collection_id = env::var("COLLECTION_ID").unwrap_or_else(|_| "smart-door-faces".to_string());
        let esp32_cam_url = env::var("ESP32_CAM_CAPTURE_URL").unwrap_or_else(|_| "http://192.168.1.140/capture".to_string());
        let pico2_door_url = env::var("PICO2_DOOR_URL").unwrap_or_else(|_| "http://192.168.1.141/door".to_string());
        let confidence_threshold = env::var("CONFIDENCE_THRESHOLD")
            .unwrap_or_else(|_| "75.0".to_string())
            .parse::<f32>()
            .unwrap_or(75.0);
        
        let state = AppState {
            rekognition_client: rekognition_client.clone(),
            collection_id: collection_id.clone(),
            access_log: Arc::new(Mutex::new(Vec::new())),
            authorized_people: Arc::new(Mutex::new(HashMap::new())),
            esp32_cam_url,
            pico2_door_url,
            confidence_threshold,
        };
        
        // Initialize collection
        state.ensure_collection_exists().await?;
        state.load_existing_faces().await?;
        
        Ok(state)
    }
    
    async fn ensure_collection_exists(&self) -> Result<()> {
        info!("🔍 Checking collection '{}'...", self.collection_id);
        
        match self
            .rekognition_client
            .describe_collection()
            .collection_id(&self.collection_id)
            .send()
            .await
        {
            Ok(_) => {
                info!("✅ Collection '{}' exists", self.collection_id);
            }
            Err(_) => {
                info!("🏗️ Creating collection '{}'...", self.collection_id);
                
                self.rekognition_client
                    .create_collection()
                    .collection_id(&self.collection_id)
                    .send()
                    .await?;
                
                info!("✅ Created collection '{}'", self.collection_id);
            }
        }
        
        Ok(())
    }
    
    async fn load_existing_faces(&self) -> Result<()> {
        info!("👥 Loading existing authorized faces...");
        
        let response = self
            .rekognition_client
            .list_faces()
            .collection_id(&self.collection_id)
            .send()
            .await?;
        
        let mut people = self.authorized_people.lock().unwrap();
        
        if let Some(faces) = response.faces {
            for face in faces {
                if let (Some(face_id), Some(external_id)) = (face.face_id, face.external_image_id) {
                    let person = AuthorizedPerson {
                        name: external_id.clone(),
                        face_id: face_id.clone(),
                        external_image_id: external_id.clone(),
                        added_at: Utc::now(),
                    };
                    people.insert(face_id, person);
                }
            }
        }
        
        info!("✅ Loaded {} authorized faces", people.len());
        Ok(())
    }
    
    async fn capture_from_esp32(&self) -> Result<Bytes> {
        info!("📸 Capturing image from ESP32-CAM at {}", self.esp32_cam_url);
        
        let response = reqwest::get(&self.esp32_cam_url).await?;
        
        if response.status().is_success() {
            let image_data = response.bytes().await?;
            info!("✅ Captured {} bytes from ESP32-CAM", image_data.len());
            Ok(image_data)
        } else {
            Err(anyhow::anyhow!("ESP32-CAM capture failed: {}", response.status()))
        }
    }
    
    async fn control_pico2_door(&self, unlock: bool) -> Result<()> {
        let action = if unlock { "unlock" } else { "lock" };
        info!("🚪 Sending {} command to Pico 2", action);
        
        let payload = serde_json::json!({
            "action": action,
            "timestamp": Utc::now().timestamp()
        });
        
        let client = reqwest::Client::new();
        let response = client
            .post(&self.pico2_door_url)
            .json(&payload)
            .send()
            .await?;
        
        if response.status().is_success() {
            info!("✅ Pico 2 door {} successful", action);
        } else {
            warn!("⚠️ Pico 2 door {} failed: {}", action, response.status());
        }
        
        Ok(())
    }
    
    async fn add_person(&self, name: String, image_data: Bytes) -> Result<AddPersonResponse> {
        info!("➕ Adding person '{}' to collection", name);
        
        let image = Image::builder()
            .bytes(image_data.to_vec().into())
            .build();
        
        let response = self
            .rekognition_client
            .index_faces()
            .collection_id(&self.collection_id)
            .image(image)
            .external_image_id(&name)
            .max_faces(1)
            .quality_filter(QualityFilter::Auto)
            .send()
            .await?;
        
        if let Some(face_records) = response.face_records {
            if let Some(face_record) = face_records.first() {
                if let Some(face) = &face_record.face {
                    if let Some(face_id) = &face.face_id {
                        let person = AuthorizedPerson {
                            name: name.clone(),
                            face_id: face_id.clone(),
                            external_image_id: name.clone(),
                            added_at: Utc::now(),
                        };
                        
                        self.authorized_people
                            .lock()
                            .unwrap()
                            .insert(face_id.clone(), person);
                        
                        self.log_access(
                            format!("➕ Added authorized person: {}", name),
                            Some(name.clone()),
                            None,
                            false,
                        );
                        
                        return Ok(AddPersonResponse {
                            face_id: face_id.clone(),
                            message: format!("✅ Successfully added {}", name),
                        });
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("No face detected in image"))
    }
    
    async fn recognize_face(&self, image_data: Bytes) -> Result<AccessCheckResponse> {
        info!("🔍 Attempting face recognition...");
        
        let image = Image::builder()
            .bytes(image_data.to_vec().into())
            .build();
        
        let response = self
            .rekognition_client
            .search_faces_by_image()
            .collection_id(&self.collection_id)
            .image(image)
            .max_faces(1)
            .face_match_threshold(self.confidence_threshold)
            .send()
            .await?;
        
        let timestamp = Utc::now();
        
        if let Some(face_matches) = response.face_matches {
            if let Some(face_match) = face_matches.first() {
                if let (Some(face), Some(similarity)) = (&face_match.face, face_match.similarity) {
                    if let Some(external_id) = &face.external_image_id {
                        let confidence = similarity / 100.0;
                        
                        // Control door
                        if let Err(e) = self.control_pico2_door(true).await {
                            warn!("Failed to unlock door: {}", e);
                        }
                        
                        self.log_access(
                            format!("🟢 Access GRANTED - {}", external_id),
                            Some(external_id.clone()),
                            Some(confidence),
                            true,
                        );
                        
                        return Ok(AccessCheckResponse {
                            access_granted: true,
                            person_name: Some(external_id.clone()),
                            confidence: Some(confidence),
                            timestamp,
                        });
                    }
                }
            }
        }
        
        self.log_access(
            "🔴 Access DENIED - Face not recognized".to_string(),
            None,
            None,
            false,
        );
        
        Ok(AccessCheckResponse {
            access_granted: false,
            person_name: None,
            confidence: None,
            timestamp,
        })
    }
    
    fn log_access(&self, action: String, person_name: Option<String>, confidence: Option<f32>, access_granted: bool) {
        let log_entry = AccessLog {
            timestamp: Utc::now(),
            action: action.clone(),
            person_name,
            confidence,
            access_granted,
        };
        
        self.access_log.lock().unwrap().push(log_entry);
        info!("📝 {}", action);
    }
    
    fn get_recent_logs(&self, limit: usize) -> Vec<AccessLog> {
        let logs = self.access_log.lock().unwrap();
        logs.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    fn get_authorized_people(&self) -> Vec<String> {
        self.authorized_people
            .lock()
            .unwrap()
            .values()
            .map(|p| p.name.clone())
            .collect()
    }
}

// Web handlers
async fn dashboard(State(state): State<AppState>) -> Html<String> {
    let logs = state.get_recent_logs(10);
    let people = state.get_authorized_people();
    
    let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>🦀 Smart Door Lock - Rust + AWS</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ 
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            padding: 25px; margin: 20px 0; border-radius: 16px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);
        }}
        .status {{ 
            padding: 20px; margin: 15px 0; border-radius: 12px; 
            border-left: 5px solid;
        }}
        .success {{ 
            background: linear-gradient(135deg, #d4edda, #c3e6cb); 
            color: #155724; border-left-color: #28a745; 
        }}
        .info {{ 
            background: linear-gradient(135deg, #d1ecf1, #bee5eb); 
            color: #0c5460; border-left-color: #17a2b8; 
        }}
        .warning {{ 
            background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
            color: #856404; border-left-color: #ffc107; 
        }}
        button {{ 
            padding: 12px 24px; margin: 8px; border: none; border-radius: 8px; 
            cursor: pointer; font-weight: 600; font-size: 14px;
            transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .btn-primary {{ background: linear-gradient(135deg, #007bff, #0056b3); color: white; }}
        .btn-success {{ background: linear-gradient(135deg, #28a745, #1e7e34); color: white; }}
        .btn-warning {{ background: linear-gradient(135deg, #ffc107, #e0a800); color: #212529; }}
        .btn-danger {{ background: linear-gradient(135deg, #dc3545, #c82333); color: white; }}
        button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        input[type="file"], input[type="text"] {{ 
            margin: 10px 0; padding: 12px; border: 2px solid #ddd; 
            border-radius: 8px; width: 280px; font-size: 14px;
        }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat {{ text-align: center; padding: 20px; }}
        .stat-number {{ font-size: 3em; font-weight: 800; color: #007bff; margin-bottom: 5px; }}
        .stat-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        h1 {{ 
            color: white; text-align: center; margin-bottom: 30px; font-size: 2.5em; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: 700;
        }}
        h3 {{ 
            color: #2c3e50; border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; margin-bottom: 20px; font-size: 1.2em;
        }}
        .log-entry {{ 
            padding: 12px; margin: 5px 0; border-radius: 8px;
            display: flex; justify-content: space-between; align-items: center;
            background: rgba(248, 249, 250, 0.8);
        }}
        .access-granted {{ 
            background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
            border-left: 4px solid #28a745; color: #155724; font-weight: 600;
        }}
        .access-denied {{ 
            background: linear-gradient(135deg, #f8d7da, #f1c2c7) !important;
            border-left: 4px solid #dc3545; color: #721c24; font-weight: 600;
        }}
        .rust-badge {{
            position: absolute; top: 20px; right: 20px; 
            background: linear-gradient(135deg, #ce422b, #a33622);
            color: white; padding: 8px 16px; border-radius: 20px;
            font-size: 12px; font-weight: 600; text-transform: uppercase;
        }}
        .feature-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 15px; margin: 20px 0; 
        }}
        .feature {{ 
            padding: 15px; background: rgba(255,255,255,0.7); 
            border-radius: 10px; text-align: center; 
        }}
    </style>
</head>
<body>
    <div class="rust-badge">⚡ Powered by Rust</div>
    <div class="container">
        <h1>🦀 Smart Door Lock</h1>
        
        <div class="status success">
            <h3>🎯 System Status</h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Authorized People</div>
                </div>
                <div class="stat">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Access Attempts</div>
                </div>
                <div class="stat">
                    <div class="stat-number">AWS</div>
                    <div class="stat-label">Rekognition</div>
                </div>
            </div>
        </div>
        
        <div class="feature-grid">
            <div class="feature">
                <h4>🚀 High Performance</h4>
                <p>Rust's zero-cost abstractions for maximum speed</p>
            </div>
            <div class="feature">
                <h4>🔒 Memory Safe</h4>
                <p>No buffer overflows or memory leaks</p>
            </div>
            <div class="feature">
                <h4>☁️ AWS Powered</h4>
                <p>Enterprise-grade face recognition</p>
            </div>
            <div class="feature">
                <h4>🔗 IoT Ready</h4>
                <p>ESP32-CAM + Pico 2 integration</p>
            </div>
        </div>
        
        <div class="card">
            <h3>➕ Add Authorized Person</h3>
            <input type="text" id="person-name" placeholder="Enter person name">
            <input type="file" id="face-photo" accept="image/*">
            <button class="btn-success" onclick="addPerson()">Add Person</button>
        </div>
        
        <div class="card">
            <h3>🔍 Access Control</h3>
            <button class="btn-primary" onclick="checkAccessESP32()">📸 Check Access (ESP32-CAM)</button>
            <input type="file" id="test-photo" accept="image/*" style="display: inline-block; width: 200px;">
            <button class="btn-warning" onclick="testAccessUpload()">🧪 Test Upload</button>
            <button class="btn-info" onclick="listPeople()">👥 List People</button>
        </div>
        
        <div class="card">
            <h3>📋 Recent Access Log</h3>
            <div id="log">
                {}
            </div>
        </div>
        
        <div class="status info">
            <h3>🔗 Hardware Integration</h3>
            <p><strong>ESP32-CAM:</strong> Captures images automatically</p>
            <p><strong>Pico 2 (Rust):</strong> Controls door lock mechanism</p>
            <p><strong>Current Mode:</strong> Manual testing + Hardware ready</p>
        </div>
    </div>
    
    <script>
        async function addPerson() {{
            const name = document.getElementById('person-name').value;
            const fileInput = document.getElementById('face-photo');
            
            if (!name || !fileInput.files[0]) {{
                alert('Please enter name and select a photo');
                return;
            }}
            
            const formData = new FormData();
            formData.append('name', name);
            formData.append('photo', fileInput.files[0]);
            
            try {{
                const response = await fetch('/api/add-person', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    alert('✅ ' + data.data.message);
                    location.reload();
                }} else {{
                    alert('❌ Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('❌ Network error: ' + error.message);
            }}
        }}
        
        async function checkAccessESP32() {{
            try {{
                const response = await fetch('/api/check-access-esp32', {{
                    method: 'POST'
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    const result = data.data.access_granted ? '🟢 ACCESS GRANTED' : '🔴 ACCESS DENIED';
                    const person = data.data.person_name || 'Unknown';
                    const confidence = data.data.confidence ? Math.round(data.data.confidence * 100) + '%' : 'N/A';
                    
                    alert(`${{result}}\\n\\nPerson: ${{person}}\\nConfidence: ${{confidence}}`);
                    location.reload();
                }} else {{
                    alert('❌ Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('❌ Network error: ' + error.message);
            }}
        }}
        
        async function testAccessUpload() {{
            const fileInput = document.getElementById('test-photo');
            
            if (!fileInput.files[0]) {{
                alert('Please select a photo to test');
                return;
            }}
            
            const formData = new FormData();
            formData.append('photo', fileInput.files[0]);
            
            try {{
                const response = await fetch('/api/check-access', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    const result = data.data.access_granted ? '🟢 ACCESS GRANTED' : '🔴 ACCESS DENIED';
                    const person = data.data.person_name || 'Unknown';
                    const confidence = data.data.confidence ? Math.round(data.data.confidence * 100) + '%' : 'N/A';
                    
                    alert(`${{result}}\\n\\nPerson: ${{person}}\\nConfidence: ${{confidence}}`);
                    location.reload();
                }} else {{
                    alert('❌ Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('❌ Network error: ' + error.message);
            }}
        }}
        
        async function listPeople() {{
            try {{
                const response = await fetch('/api/list-people');
                const data = await response.json();
                
                if (data.success && data.data.length > 0) {{
                    const people = data.data.join('\\n• ');
                    alert(`👥 Authorized People (${{data.data.length}})::\\n\\n• ${{people}}`);
                }} else {{
                    alert('👥 No authorized people found\\n\\nAdd someone using the form above!');
                }}
            }} catch (error) {{
                alert('❌ Network error: ' + error.message);
            }}
        }}
    </script>
</body>
</html>
    "#, 
    people.len(),
    logs.len(),
    logs.iter()
        .map(|log| {
            let status_class = if log.access_granted { "access-granted" } else { "access-denied" };
            let confidence = log.confidence
                .map(|c| format!(" ({}%)", (c * 100.0) as i32))
                .unwrap_or_default();
            
            format!(
                r#"<div class="log-entry {}">
                    <span><strong>{}</strong> - {}</span>
                    <span>{}</span>
                </div>"#,
                status_class,
                log.timestamp.format("%m-%d %H:%M:%S"),
                log.action,
                confidence
            )
        })
        .collect::<Vec<_>>()
        .join("")
    );
    
    Html(html)
}

async fn add_person_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<AddPersonResponse>>, StatusCode> {
    let mut name = None;
    let mut image_data = None;
    
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let field_name = field.name().unwrap_or("");
        
        match field_name {
            "name" => {
                name = Some(field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            "photo" => {
                image_data = Some(field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            _ => {}
        }
    }
    
    let name = name.ok_or(StatusCode::BAD_REQUEST)?;
    let image_data = image_data.ok_or(StatusCode::BAD_REQUEST)?;
    
    match state.add_person(name, image_data).await {
        Ok(response) => Ok(Json(ApiResponse {
            success: true,
            data: Some(response),
            error: None,
        })),
        Err(e) => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(e.to_string()),
        })),
    }
}

async fn check_access_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<AccessCheckResponse>>, StatusCode> {
    let mut image_data = None;
    
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        if field.name() == Some("photo") {
            image_data = Some(field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            break;
        }
    }
    
    let image_data = image_data.ok_or(StatusCode::BAD_REQUEST)?;
    
    match state.recognize_face(image_data).await {
        Ok(response) => Ok(Json(ApiResponse {
            success: true,
            data: Some(response),
            error: None,
        })),
        Err(e) => Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some(e.to_string()),
        })),
    }
}

async fn check_access_esp32_handler(
    State(state): State<AppState>,
) -> Json<ApiResponse<AccessCheckResponse>> {
    match state.capture_from_esp32().await {
        Ok(image_data) => {
            match state.recognize_face(image_data).await {
                Ok(response) => Json(ApiResponse {
                    success: true,
                    data: Some(response),
                    error: None,
                }),
                Err(e) => Json(ApiResponse {
                    success: false,
                    data: None,
                    error: Some(e.to_string()),
                }),
            }
        }
        Err(e) => Json(ApiResponse {
            success: false,
            data: None,
            error: Some(format!("ESP32-CAM capture failed: {}", e)),
        }),
    }
}

async fn list_people_handler(State(state): State<AppState>) -> Json<ApiResponse<Vec<String>>> {
    let people = state.get_authorized_people();
    Json(ApiResponse {
        success: true,
        data: Some(people),
        error: None,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let state = AppState::new().await?;
    
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/api/add-person", post(add_person_handler))
        .route("/api/check-access", post(check_access_handler))
        .route("/api/check-access-esp32", post(check_access_esp32_handler))
        .route("/api/list-people", get(list_people_handler))
        .layer(tower::ServiceBuilder::new()
            .layer(tower_http::limit::RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
            .layer(CorsLayer::permissive())
        )
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    
    info!("🦀 Smart Door Lock server running on http://localhost:3000");
    info!("🔒 High-performance Rust + AWS Rekognition");
    info!("🔗 ESP32-CAM + Pico 2 integration ready");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}