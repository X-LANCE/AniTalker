import os
import sys
sys.path.append("./")
import gradio as gr
import cv2
import shutil
import uuid
import tempfile
import numpy as np
import soundfile as sf
import subprocess
import pandas as pd
from torch import multiprocessing as mp
import time
from animate_portrait_sr import animate_portrait
#animate_portrait(img_path, audio_path,outfile_path)
#要求实现的接口，input:img_path,audio_path,outfile_path
#pip install gradio == 3.40.1

# from src.constant import settings
# from src.constant import AvatarID
# from src.model.utils.video_process import (
#     merge_video, save_images_for_video)
# from src.constant import FaceswapType
# from src.service.avatar_infer_v2 import (
#     AvatarInferService, AvatarRegisterService)

demo_async = gr.Blocks()

# Basic set up
__cache_dir__ = "demo/avatar_cache"
if not os.path.exists(__cache_dir__):
    os.mkdir(__cache_dir__)

__temp_dir__ = "demo/avatar_cache/temp"
if not os.path.exists(__temp_dir__):
    os.mkdir(__temp_dir__)


__service_infer__ = None
__service_register__ = None
# mp.set_start_method(method='spawn', force=True)
__session_caches_queue__ = mp.Queue()
__avatars__ = []
__processed_tasks__ = []
__to_process_tasks__ = []
__session_cachesss__ = []
session_caches_async = []


def enqueue(audio, video, avatar_name_register, task_type, face_swap_type):
    print("AA:;audio:::", audio)
    print("AA:;audio:::", video)
    print("AA:;audio:::", avatar_name_register)
    # print("AA:;audio:::", session_caches)
    print("AA:;audio:::", task_type)
    
    global __session_caches_queue__
    global __to_process_tasks__
    global __session_cachesss__
    session_caches = __session_cachesss__
    session_caches = [{'avatar_name': '单图驱动任务'}]
    
    if task_type == "register":
        if len(session_caches) == 0:
            session_caches.append(
                {"avatar_name": avatar_name_register})
        else:
            session_caches[0]["avatar_name"] = avatar_name_register

    my_uuid = str(uuid.uuid4())
    print("AAaaa:::::::video:::", audio, video, session_caches, my_uuid,)
    __session_caches_queue__.put(
        (audio, video, session_caches, my_uuid,
         task_type, face_swap_type))

    __to_process_tasks__.append({
        "avatar_name": session_caches[0]["avatar_name"],
        "my_uuid": my_uuid,
        "task_type": task_type,
        "status": "todo"
    })

    return my_uuid, update_tasks()


def dequeue_delete():
    global __session_caches_queue__
    global __to_process_tasks__

    """Delete rubbish task"""
    __session_caches_queue__.get()
    del __to_process_tasks__[0]

    return update_tasks()


def dequeue_loop(
        q, avatars, processed_tasks, to_process_tasks, cache_dir, temp_dir):
    # service_infer, service_register = initialization(avatars, cache_dir)

    # DEBUG code, keep for pdb access.
    # session_caches = [{
    #     "avatar_id": AvatarID(
    #         avatar_name="test26", video_name="jianhao_test_short")
    #     }]
    # synthesize(
    #     "/tmp/silence.wav", session_caches, service_infer,
    #     cache_dir, temp_dir, "test")
    # DEBUG code, keep for pdb access.

    while True:
        time.sleep(0.1)
        if not q.empty():
            audio, video, session_caches, my_uuid, task_type, face_swap_type \
                = q.get()

            del to_process_tasks[0]
            processed_task = {
                "avatar_name": session_caches[0]["avatar_name"],
                "my_uuid": my_uuid,
                "task_type": task_type,
                "status": "processing"
            }
            processed_tasks.append(processed_task)

            try:
                if task_type == "synthesize":
                    print("Synthesizing:")
                    print(session_caches)
                    synthesize(
                        audio, video, session_caches,
                        cache_dir, temp_dir, my_uuid=my_uuid)
                # elif task_type == "register":
                #     print("Registering:")
                #     print(session_caches)
                #     register(
                #         upload_video=video,
                #         avatar_name=session_caches[0]["avatar_name"],
                #         avatars=avatars,
                #         service_infer=service_infer,
                #         service_register=service_register,
                #         cache_dir=cache_dir,
                #         face_swap_type=face_swap_type
                #     )
                processed_task["status"] = "success"
                del processed_tasks[-1]
                processed_tasks.append(processed_task)
            except Exception as e:
                print(f"{my_uuid} task FAIL.")
                print(e)
                processed_task["status"] = "fail"
                del processed_tasks[-1]
                processed_tasks.append(processed_task)


def get_video(my_uuid):
    video_path = os.path.join(__temp_dir__, my_uuid+".mp4")

    if os.path.exists(video_path):
        return video_path
    else:
        return None


def update_avatars(audio_upload_async=None,image_upload_async=None):
    print("HHHHHH::::", "nihao")

    # return [[image_upload_async, "aa_bb", ""]]
    global __avatars__
    print("AAA:::", [a for a in __avatars__])
    
    
    # avatar_list_async = update_avatars()
    global session_caches_async
    session_caches_async, avatar_name_selected_async = select_avatar(0, session_caches_async)
    
    return [a for a in __avatars__]


def update_tasks():
    global __to_process_tasks__
    global __processed_tasks__
    tasks_list = []

    for task in __to_process_tasks__:
        # headers: avatar_name, task_type, task_id, status
        tasks_list.append([
            task["avatar_name"],
            task["task_type"],
            task["my_uuid"],
            task["status"]
        ])

    # Iterate reversely, make the latest the upper positions.
    for task in __processed_tasks__[::-1]:
        # headers: avatar_name, task_type, task_id, status
        tasks_list.append([
            task["avatar_name"],
            task["task_type"],
            task["my_uuid"],
            task["status"]]
        )

    return tasks_list


def clear_tasks():
    global __processed_tasks__
    processed_tasks_new = []

    for task in __processed_tasks__:
        if task["status"] == "success" or task["status"] == "fail":
            continue
        else:
            processed_tasks_new.append(task)

    # clear
    for i in range(len(__processed_tasks__)):
        del __processed_tasks__[0]

    # Readd
    for task in processed_tasks_new:
        __processed_tasks__.append(task)

    return update_tasks()


# def initialization(avatars, cache_dir):
#     service_infer = AvatarInferService()
#     service_register = AvatarRegisterService()

#     # Default list of demo avatars.
#     default_avatar_list = init_local_list(cache_dir)
#     for avatar_id in default_avatar_list:
#         service_infer.fetch_cos(avatar_id)

#     # Local list of demo avatars in folder.
#     update_list = []
#     avatar_models_dir = os.path.join(os.path.dirname(os.path.dirname(
#         os.path.abspath(__file__))), "ModelData", "trainedmodels")
#     avatar_dirs = os.listdir(avatar_models_dir)
#     for avatar_dir in avatar_dirs:
#         fn_list = os.listdir(os.path.join(avatar_models_dir, avatar_dir))
#         mp4_fn_list = [fn for fn in fn_list if fn.endswith(".mp4")]
#         for mp4_fn in mp4_fn_list:
#             basename = mp4_fn.split(".")[0]
#             if os.path.exists(os.path.join(
#                 avatar_models_dir, avatar_dir, basename+".pkl")) and \
#             os.path.exists(os.path.join(
#                 avatar_models_dir, avatar_dir, basename+".yaml")):
#                 avatar_id = AvatarID(
#                         avatar_name=avatar_dir,
#                         video_name=basename.replace(avatar_dir+"_", "", 1))

#                 # Skip already existed default avatars.
#                 existed = False
#                 for default_avatar_id in default_avatar_list:
#                     if default_avatar_id == avatar_id:
#                         existed = True
#                         break

#                 if not existed:
#                     update_local_list(cache_dir, avatar_id)

#     avatars += read_local_list(cache_dir)

#     print("-------------Available Avatars-------------")
#     print(avatars)

#     return service_infer, service_register  # , service_register

# # 创建一个目录来保存上传的文件
# os.makedirs('uploaded_files', exist_ok=True)

# def save_file(file_path):
#     # 生成一个新的文件路径
#     new_file_path = os.path.join('uploaded_files', os.path.basename(file_path))
#     # 将文件复制到新的位置
#     shutil.copyfile(file_path, new_file_path)
#     # 返回新的文件路径
#     return new_file_path


def register(upload_video, avatar_name, avatars,
             service_infer, service_register,
             cache_dir, face_swap_type):
    # service_register = AvatarRegisterService()

    # To prevent GPU exloration,
    # while both register and inference exit
    del service_infer.worker

    do_register = service_register.register(
        upload_video=upload_video,
        avatar_name=avatar_name,
        face_swap_type=face_swap_type)
    upload_video = upload_video

    # avatar_id = AvatarID(
    #     avatar_name=avatar_name,
    #     video_name=os.path.basename(upload_video).split(".")[0])
    avatar_id = "jianhao"

    if do_register:
        # TODO: we no longer need the cropxxxx path
        avatars.append([
            upload_video, avatar_id.val, avatar_id])
        update_local_list(cache_dir, avatar_id)

    service_infer.switch_avatar_worker(avatar_id_new=avatar_id)

# def register_proc(
#         upload_video, avatar_name, avatars,
#         cache_dir, face_swap_type
# ):
#     p_register = mp.Process(
#         target=register,
#         args=(
#             upload_video, avatar_name, avatars,
#             cache_dir, face_swap_type
#         ))
#     p_register.start()
#     p_register.join()
#     p_register.terminate()


def init_local_list(cache_dir):
    record_file = os.path.join(cache_dir, "avatar_id.list")
    avatar_id_list = []

    if os.path.exists(record_file):
        print("| avatar_id.list Exists, read list.")
        with open(record_file, "r") as f:
            for line in f:
                line = line.strip()
                avatar_name, video_name = line.split(",")
                # avatar_id_list.append(
                #     AvatarID(avatar_name, video_name)
                # )
                avatar_id_list.append(
                    "jianhao"
                )

    return avatar_id_list


def update_local_list(cache_dir, avatar_id):
    record_file = os.path.join(cache_dir, "avatar_id.list")
    with open(record_file, "a") as f:
        f.write(
            f"{avatar_id.avatar_name},{avatar_id.video_name}\n")


# def read_local_list(cache_dir):
#     avatar_id_list = []
#     record_file = os.path.join(cache_dir, "avatar_id.list")

#     if not os.path.exists(record_file):
#         return []

#     with open(record_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             avatar_name, video_name = line.split(",")
#             avatar_id = AvatarID(
#                 avatar_name=avatar_name, video_name=video_name)
#             root_dir = os.path.dirname(os.path.abspath(__file__))
#             video_path = os.path.join(
#                 root_dir, "ModelData/trainedmodels/",
#                 avatar_name, f"{video_name}.mp4")
#             avatar_id_list.append([video_path, avatar_id.val, avatar_id])

#     return avatar_id_list

def render(img_path, audio_path,outfile_path):
    # video_path = os.path.join(__output_directory__, f"{uuid.uuid4()}.mp4")
    assert os.path.exists(img_path)
    assert os.path.exists(audio_path)
    # ffmpeg_cmd = f'ffmpeg -loop 1 -i "{img_path}" -i "{audio_path}" -c:v libx264 -c:a aac -strict experimental -b:a 192k -shortest "{outfile_path}"'
    # os.system(ffmpeg_cmd)
    animate_portrait(img_path, audio_path,outfile_path)
    print(f"Video saved to {outfile_path}")


def synthesize(
        audio, video, session_caches,
        cache_dir, temp_dir, my_uuid=None):
    
    
    # avatar_id = session_caches[0]["avatar_id"]

    # Prepare files.
    my_uuid = str(uuid.uuid4()) if my_uuid is None else my_uuid
    outfile_path = os.path.join(temp_dir, my_uuid+".mp4")
    temp_dir = 'tempfile_of_{}'.format(temp_dir.split('/')[-1])

    render(video, audio, outfile_path)
    
    # v_pathes_np = []
    # start_frame = 0
    # iters = service_infer.generate_avatar_np(
    #     avatar_id=avatar_id,
    #     audio_path=audio,
    #     start_frame=start_frame)

    # # Synth
    # for i, (status, v_path_np, frame_index_id) in enumerate(iters):
    #     if status == 'Success':
    #         v_pathes_np += [v_np for v_np in v_path_np]
    #     if status == 'End':
    #         start_frame = frame_index_id
    #     print(status)

    # # To video
    # with tempfile.TemporaryDirectory() as temp_png_dir:
    #     save_images_for_video(output_dir=temp_png_dir, images=v_pathes_np)
    #     os.system(
    #         f"ffmpeg -i {temp_png_dir}/%d.png -i {audio} -c:v libx264 "
    #         "-c:a aac -profile:v high -strict -2 -crf 18 -pix_fmt yuv420p -shortest"
    #         f" -y {outfile_path}")

    return outfile_path


def select_avatar(avatar_index, session_caches):
    print("KKK::::avatar_index::",avatar_index)
    print("KKK::::session_caches::",session_caches)
    global __avatars__
    session_caches = [{"avatar_name":'aa_bb', "avatar_id":'aa_bb'}]
    if len(session_caches) == 0:
        session_caches.append(
            {"avatar_id": __avatars__[avatar_index][2],
             "avatar_name": __avatars__[avatar_index][1]})
    else:
        # import pdb; pdb.set_trace()
        avatar_index=0
        # session_caches[0]["avatar_id"] = __avatars__[avatar_index][2]
        # session_caches[0]["avatar_name"] = __avatars__[avatar_index][1]
        session_caches[0]["avatar_id"] = "单图驱动数字人"
        session_caches[0]["avatar_name"] = "单图驱动数字人"
    __session_cachesss__ = session_caches
    return session_caches, session_caches[0]["avatar_id"]
    # return 
    # return [{"avatar_name":'aa_bb'}], 'aa_bb'


with demo_async:
    gr.Markdown("## 单图驱动数字人demo视频生产平台-异步任务提交")

    session_caches_async = gr.State(
        [{
            "avatar_name": "单图驱动任务",
        }])
    
    
    type_synthesize = gr.Textbox(
        value="synthesize",
        interactive=False,
        visible=False)
    type_register = gr.Textbox(
        value="register",
        interactive=False,
        visible=False)

    with gr.Box():
        with gr.Row():
            # Register
            # with gr.Column(scale=1):
            #     label_register = gr.Label(value="注册新的Avatar", label="")

            #     upload_video = gr.Video(
            #         label='step1: 上传口播视频注册avatar', interactive=True,
            #         height=720)

            #     with gr.Row():
            #         avatar_name_register = gr.Textbox(
            #             label="step2: 输入avatar name",
            #             show_label=False,
            #             placeholder="step2: 输入 avatar name",
            #             interactive=True,
            #         ).style(
            #             border=(True, False, True, True),
            #             rounded=(True, False, False, True),
            #             container=False,
            #         )

            #     radio_face_swap = gr.Radio(
            #         [FaceswapType.FACE_NECK, FaceswapType.FACE_ONLY],
            #         label="注册换脸模式 (通常使用face_only，若下巴/脖子变化效果不好改用face_neck)",
            #         value=FaceswapType.FACE_ONLY
            #     )

            #     btn_register = gr.Button(value="提交异步注册任务")

            # Synthesize
        
            with gr.Column(scale=1):
                upload_video_hidden_async = gr.Video(
                    label='source_video', visible=False)

                avatar_name_hidden_async = gr.Textbox(
                    label="avatar_name",
                    show_label=False,
                    placeholder="avatar name",
                    interactive=True,
                    visible=False
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )

                label_synthesize_async = gr.Label(
                    value="单图驱动", label="")
                
        with gr.Row():
            with gr.Column(scale=1):
                # upload audio
                audio_upload_async = gr.Audio(
                    source="upload", label="step1: 上传wav文件",
                    show_label=True, interactive=True, type="filepath")
            with gr.Column(scale=1):   
                image_upload_async = gr.Image(
                    source="upload", label='step2: 上传单图驱动图片', show_label=True, interactive=True, type="filepath",
                    height=256)

        with gr.Row():    # select avatar
                btn_update_avatars = gr.Button(
                    value="step3: 🚩必须先点击一下此处后再提交异步任务！！！")

                avatar_list_async = gr.Dataset(
                    label="step2: 选择avatar",
                    samples=[a for a in __avatars__],
                    components=[upload_video_hidden_async, avatar_name_hidden_async],
                    type="index", interactive=True, visible=False)

                avatar_name_selected_async = gr.Textbox(
                    label="avatar selected",
                    show_label=False,
                    placeholder="avatar selected",
                    interactive=False,visible=False
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
        with gr.Row():
                btn_synthesize_async_upload = gr.Button(value="step4: 🚀提交异步生成任务")

        with gr.Row():
            gr.Markdown("#### 任务ID (请记住任务ID，用于查询状态和获取文件):")

        with gr.Row():
            task_id_display = gr.Textbox(
                # label="任务ID (请记住任务ID，用于查询状态和获取文件)",
                show_label=False,
                placeholder="task_id",
                interactive=True,
                visible=True
            ).style(
                border=(True, False, True, True),
                rounded=(True, False, False, True),
                container=False,
            )

        tasks_data = gr.Dataframe(
            headers=[
                "avatar_name",
                "task_type", "task_id", "status"],
            datatype=["str", "str", "str", "str"],
            # row_count=50,
            col_count=(4, "fixed"),
            interactive=True
        )


    gr.Markdown("## 单图驱动数字人demo视频生产平台-查询/删除任务")

    with gr.Row():
        with gr.Column():
            btn_check_tasks = gr.Button(value="查询所有任务情况")

            btn_delete_task = gr.Button(
                value="[删除] 谨慎使用: 删除一个todo任务(越早添加的任务，越先被删除)")

            btn_clear_tasks = gr.Button(
                value="[清理] 清理所有success/fail任务记录")

    gr.Markdown("## 单图驱动数字人demo视频生产平台-获取视频")

    with gr.Row():
        with gr.Column():
            task_id_check = gr.Textbox(
                label="请输入任务ID获取生成文件",
                show_label=False,
                placeholder="task_id",
                interactive=True,
                visible=True
            ).style(
                border=(True, False, True, True),
                rounded=(True, False, False, True),
                container=False,
            )

            btn_synthesize_async_get = gr.Button(value="查询和提取文件")

            output_video_async = gr.File(
                file_types=["mp4"], label='输出视频', width=480)

        btn_check_tasks.click(
            update_tasks, inputs=[], outputs=[tasks_data])

        btn_clear_tasks.click(
            clear_tasks, inputs=[], outputs=[tasks_data])

        btn_synthesize_async_get.click(
            get_video,
            inputs=[task_id_check],
            outputs=[output_video_async]
        )

        btn_delete_task.click(dequeue_delete, inputs=[], outputs=[tasks_data])

        avatar_list_async.click(
            select_avatar,
            inputs=[avatar_list_async, session_caches_async],
            outputs=[session_caches_async, avatar_name_selected_async])
      
        btn_update_avatars.click(
            update_avatars, inputs=[], outputs=[avatar_list_async])
        # audio_upload_async,image_upload_async
        # btn_register.click(
        #     enqueue,
        #     inputs=[audio_upload_async, image_upload_async,
        #             avatar_name_register, session_caches_async,
        #             type_register, radio_face_swap],
        #     outputs=[task_id_display, tasks_data])
        # print("KKK::session_caches_async::", session_caches_async)
        avatar_name_register = gr.Textbox(value="单图驱动数字人", visible=False)
        btn_synthesize_async_upload.click(
            enqueue,
            inputs=[audio_upload_async, image_upload_async, avatar_name_register,
                    type_synthesize],
            outputs=[task_id_display, tasks_data]
        )


def main():
    global __avatars__
    global __session_caches_queue__
    global __processed_tasks__
    global __to_process_tasks__

    __mq_manager__ = mp.Manager()
    __avatars__ = __mq_manager__.list()
    __processed_tasks__ = __mq_manager__.list()
    __to_process_tasks__ = __mq_manager__.list()

    p_loop = mp.Process(
        target=dequeue_loop,
        args=(
            __session_caches_queue__,
            __avatars__,
            __processed_tasks__,
            __to_process_tasks__,
            __cache_dir__,
            __temp_dir__,
        ))

    return p_loop


def main_debug():
    global __avatars__
    global __session_caches_queue__
    global __processed_tasks__
    global __to_process_tasks__

    __mq_manager__ = mp.Manager()
    __avatars__ = __mq_manager__.list()
    __processed_tasks__ = __mq_manager__.list()
    __to_process_tasks__ = __mq_manager__.list()

    dequeue_loop(
            __session_caches_queue__,
            __avatars__,
            __processed_tasks__,
            __to_process_tasks__,
            __cache_dir__,
            __temp_dir__,
    )


if __name__ == "__main__":
    # default port: 7860
    p_loop = main()
    p_loop.start()

    # DEBUG code, keep for pdb access.
    # for i in range(5):
    #     time.sleep(1)
    #     print(f"wait 1 second")
    # print("execute main_debug")
    # main_debug()
    # DEBUG code, keep for pdb access.

    demo = gr.TabbedInterface(
        [demo_async],
        ["提交异步任务"])
    demo.launch(server_name="0.0.0.0", server_port=7860)

    p_loop.join()
