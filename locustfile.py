import gevent
import requests
import time
import os
import locust
from locust import HttpUser, task, constant

def async_success(name, start_time, resp):
    locust.events.request_success.fire(
        request_type=resp.request.method,
        name=name,
        response_time=int((time.monotonic() - start_time) * 1000),
        response_length=len(resp.content),
    )

def async_failure(name, start_time, resp, message):
    locust.events.request_failure.fire(
        request_type=resp.request.method,
        name=name,
        response_time=int((time.monotonic() - start_time) * 1000),
        exception=Exception(message),
    )

class loadTestTrain(HttpUser):

    wait_time = constant(1)

    def _do_async_thing_handler(self, timeout=600):
        test_data = {
            "dataset_filename": "5507a85a-296f-4118-a266-773ceb6ad84b.csv",
            "label_columns": ["Rainfall"],
            "target_column": "RainTomorrow",
            "model_type": "Decision Tree"
        }
        post_resp = requests.post(os.path.join(self.host, 'train'), json=test_data)
        if not post_resp.status_code == 202:
            return
        task_id = post_resp.json()['task_id']
        print(task_id)

        # Now poll for an ACTIVE status
        start_time = time.monotonic()
        end_time = start_time + timeout
        while time.monotonic() < end_time:
            r = requests.get(os.path.join(self.host, 'train/') + task_id)
            if r.status_code == 200 and r.json()['accuracy'] != None:
                async_success('POST /train/task_id - async', start_time, post_resp)
                return

            # IMPORTANT: Sleep must be monkey-patched by gevent (typical), or else
            # use gevent.sleep to avoid blocking the world.
            time.sleep(1)
        async_failure('POST /train/task_id - async', start_time, post_resp,
                      'Failed - timed out after %s seconds' % timeout)

    @task
    def do_async_thing(self):
        gevent.spawn(self._do_async_thing_handler)
