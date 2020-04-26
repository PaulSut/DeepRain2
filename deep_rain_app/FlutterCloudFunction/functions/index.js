const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp(functions.config().function);

var newData;

exports.pushNotificationTrigger = functions.firestore.document('RainWarningPushNotification/{messageId}').onCreate(async (snapshot, context) =>{
    if(snapshot.isEmpty){
        console.log('No Device');
        return;
    }

    //the data from the document in RainWarningPushNotification (which the server pushed to trigger a push notification)
    newData = snapshot.data();

    //get all device tokens for the region where it is rainy
    const snap_region = await admin.firestore().collection('Regions/' + newData.region + '/tokens').get();
    const tokens_in_region = [];
    var index_region = 0;
    snap_region.forEach(doc => {
        tokens_in_region[index_region] = doc.data().token;
        index_region = index_region + 1;
    });
    console.log('tokens in region: ' + tokens_in_region)

    //get all the device tokens for the time, it will rain.
    var snap_time = await admin.firestore().collection('TimeBeforeRaining/' + newData.time_before_raining.toString()+'_min' + '/tokens').get();
    const tokens_in_time = [];
    var index_time = 0;
    snap_time.forEach(doc => {
        tokens_in_time[index_time] = doc.data().token;
        index_time = index_time + 1;
    });

    //check which device token is in this region and in the time
    //add it to tokens list, so it will get a push notification
    var tokens = [];
    for(var regionToken of tokens_in_region){
        if(tokens_in_time.includes(regionToken)){
            tokens.push(regionToken);
        }
    }

    //payload for the push notification
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


