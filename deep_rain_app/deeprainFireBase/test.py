import pyrebase

config = {
  "type": "service_account",
  "project_id": "deeprain",
  "private_key_id": "bcbc99b37e23f4d66e13dc9cbd2efe182b96860d",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDK+yS4jmMEyk/+\naldo4+A33ws15dT/5nlZtHmhv1KkYCChxKCjS40/Q8zcCb1EVUQ9FJ1xEBne+xRP\n6sVqd7wffGc6Z7E+os0vY5ouLM09OULAK5Z1X0yTa4fZP4+e2g2SRoTCZK6VeDf1\nCbSTBzm26I0vg15JeFz+0Fx7EY7MBrt766rXD9ZKHiebXBwRzGyNMMSGQZvkpQOT\nerM9Le8VEGYeFfULJJi+p23irr3Wg9V7ZhMNx2gVcFJnrpltYn0Qy/G07XOwa/VO\n1dkDvGbgvm+aCqfPAiBV4WwyGIL0uJe6l0kvwm8slyDltqmHWKGlcKw2VC2tRaX1\nKJ0A3ZOvAgMBAAECggEACveY21PuK7MWsWNjh6pQUhR+EZJyeMUmJ1+l9sRUnccH\n3FW4bDpHznRGXFk2XbvRrQ8xFovNKgwc49iGhCMmCfl5xmEV5ZF8TqQuD3KiQD2Y\nyrNQNNVKWOjFdUqG4wlbFF01DR/Ngwv00guNMyh/yM812axaatPTCn2NkaS4N3PI\n7ahfp+EiDi5QNwEuWEUtdtxkaebf5F3Knro1QQTS6h9upJ9r1xq/YXqMa1Xr2niJ\nkZXE9tksHJLyuzcu0x/dqZaPYh0KFMzts2DsqaInibuR7KPkpdYkqCnjG2pUG774\nbxAQVsNP/iDEJNB65zW8xrSzhC9v7vedtS2UwMDjSQKBgQD6WILGc3+0rcFa9K35\n6Qc68x7sp7VHC+7QBhhwGSJaZHMSMGUTwBXa+xgL65ybnEOrMTC3G6MbRWQIKpgz\npStWG8mAygrofGDjcXKnozdCJ+mVWYorBaC486c4JPs7AXk04rdvCS4TP1j0M3iR\nRSjbh1BKekb8tgN8XRzPlJLcEwKBgQDPkMWNiw1HwqcXT1FsvG7kPgwcrH+fS0Bc\nJSU+7mVgkYCSK9xeZkVKJXQ07uBz5MTAzRyqoaqg+8hhy5bK1GIvBXoXzNn/rwSZ\nZcPG/UVdswo7VIqECSkfNcxdvn9Gyb54Kw0UlA+Rrt5vYipzX7WYP/+wHJHrBsMp\ndtdkkdbldQKBgExQcv0f9EwVs52QfQBzwtp7hm/J5/7SamhLk2OIpeo8OL/QtN9C\nbOcmJ/xqRIFDL3WeNyV6bUxA6O98XI1RylYjfleeP5kDYV20Fal4nb2zAQegFfhr\nbMU0sULyMwuXtWUjv5s/hyYi53D5x8Uw+fWB2D377GvY6FMe50t4FRS1AoGBAJUK\nLPAPkn+eg4A3Ug4z16RH8UZ9jj4BOLtxaKchYuXOVHWgzOFzqfktn52KkNxmveh6\n30SeRPjHEgE2wJXvnniJBDwmubngH/tNmSA/KBm5v3UOpU9djIt+g7okWRupLPz4\nXLIUBoHjAJEV+clSRgCGo7//I7+Q4X3aeZsbJSUpAoGBAO2rYwaM17/UdSxrehwF\nyJhp5qHBHjtMzptx2NqbVk+SstFe83ehZMfV49XBJ3DQF0rUhMs7jAZTGcC6tiR+\nhQCp3vFmoSH8lRlD8eX5Z+sCTFib+dyWQRmzuZ0ClSYcVDJjexiDBVkrGKNiB5fC\nSdKS1OpBDx4BdZlMXFYwHtl8\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xpcbj@deeprain.iam.gserviceaccount.com",
  "client_id": "102246068871610197283",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xpcbj%40deeprain.iam.gserviceaccount.com"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


path_on_cloud = 'images/example.png'
path_local = 'assets/1.png'
storage.child(path_on_cloud).put(path_local)