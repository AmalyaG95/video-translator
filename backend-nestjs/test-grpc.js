const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');

// Load the proto file
const PROTO_PATH = path.join(__dirname, 'src/ml-client/proto/translation.proto');
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true
});

const translationProto = grpc.loadPackageDefinition(packageDefinition).translation;

// Create client
const client = new translationProto.TranslationService('localhost:50051', grpc.credentials.createInsecure());

// Test GetAIInsights method
console.log('Testing gRPC GetAIInsights method...');
console.log('Available methods:', Object.getOwnPropertyNames(client.__proto__));

// Try to call GetAIInsights
client.GetAIInsights({ session_id: 'test-session' }, (error, response) => {
  if (error) {
    console.error('gRPC Error:', error.message);
  } else {
    console.log('gRPC Response:', response);
  }
});
