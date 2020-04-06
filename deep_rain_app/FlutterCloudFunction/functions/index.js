const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp(functions.config().function);

var newData;

exports.pushNotificationTrigger = functions.firestore.document('RainWarningPushNotification/{messageId}').onCreate(async (snapshot, context) =>{
    if(snapshot.isEmpty){
        console.log('No Device');
        return;
    }
    var tokens = [];
    newData = snapshot.data();

    var timeBeforeRaining = newData.time_before_raining;
    var deviceTokens = await admin.firestore().collection('DeviceTokens_'+timeBeforeRaining.toString()+'_min').get();

    for(var token of deviceTokens.docs){
        tokens.push(token.data().token);
    }
    var payload = {
        notification: {title: newData.title, body: newData.body, sound: 'default', icon: 'regenschirm.png'},
        data: {click_action: 'FLUTTER_NOTIFICATION_CLICK',message: 'Push Message'}
    };
    try{
        const response = await admin.messaging().sendToDevice(tokens, payload);
        console.log('Notification sent successfully');
    }catch (err) {
        console.log('Error sending Notification');
    }
});

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });

